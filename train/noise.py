import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional
from model.GIFT import GIFT, SklearnGIFTWrapper
from model.litanfis import LitAnfis, SklearnLitAnfisWrapper
from torch.optim.lr_scheduler import OneCycleLR
import warnings

warnings.filterwarnings('ignore')

class RobustnessEvaluator:
    
    def __init__(self, 
                 experiment,
                 model_params_gift: Dict,
                 model_params_litanfis: Dict,
                 learning_params: Dict,
                 device: torch.device,
                 binary: bool = True,
                 noise_levels: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                 n_runs: int = 3,
                 random_state: int = 42):

        self.experiment = experiment
        self.model_params_gift = model_params_gift
        self.model_params_litanfis = model_params_litanfis
        self.learning_params = learning_params
        self.device = device
        self.binary = binary
        self.noise_levels = noise_levels
        self.n_runs = n_runs
        self.random_state = random_state
        
        # Store results
        self.results = {
            'gift': {nl: {'train_acc': [], 'test_acc': [], 'train_auc': [], 'test_auc': []} 
                    for nl in noise_levels},
            'litanfis': {nl: {'train_acc': [], 'test_acc': [], 'train_auc': [], 'test_auc': []} 
                        for nl in noise_levels}
        }
        
        # Get data
        self.X_train, self.y_train = self.experiment.train_numpy()
        self.X_test, self.y_test = self.experiment.test_numpy()
        
        # Convert labels to 1D if needed
        if self.y_train.ndim == 2:
            self.y_train = np.argmax(self.y_train, axis=1)
        if self.y_test.ndim == 2:
            self.y_test = np.argmax(self.y_test, axis=1)
            
    def _apply_noise(self, 
                 X: np.ndarray, 
                 noise_std: float, 
                 noise_type: str = "gaussian") -> np.ndarray:

        if noise_std == 0:
            return X
        
        # Always work on a float32 tensor internally; always return numpy
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        if noise_type == "gaussian":
            # Standard Gaussian noise - default
            noise = torch.randn_like(X_tensor) * noise_std
            result = X_tensor + noise
            
        elif noise_type == "shift":
            # Shift noise - tests boundary robustness
            shift = torch.randn(1).item() * noise_std * 2
            result = X_tensor + shift
            
        elif noise_type == "outlier":
            # Outlier noise - extreme values
            outlier_mask = torch.rand_like(X_tensor) < min(0.05 + noise_std, 0.3)
            outlier_values = torch.randn_like(X_tensor) * 10 * noise_std
            result = X_tensor.clone()
            result[outlier_mask] = outlier_values[outlier_mask]
            
        elif noise_type == "dropout":
            # Feature dropout - missing features
            dropout_prob = min(0.1 + noise_std, 0.5)
            dropout_mask = torch.rand_like(X_tensor) > dropout_prob
            result = X_tensor * dropout_mask
            
        elif noise_type == "adversarial":
            # Adversarial boundary noise - pushes across decision boundaries
            median = torch.median(X_tensor, dim=0).values
            direction = torch.where(X_tensor > median, -1.0, 1.0)
            noise = direction * torch.abs(torch.randn_like(X_tensor)) * noise_std
            result = X_tensor + noise
            
        elif noise_type == "multiplicative":
            # Multiplicative noise - scale variation
            noise = 1 + torch.randn_like(X_tensor) * noise_std
            result = X_tensor * noise
            
        else:
            raise ValueError(
                f"Unknown noise type: {noise_type}. "
                "Choose from: gaussian, shift, outlier, dropout, adversarial, multiplicative"
            )
        
        return result.numpy()
    
    def _train_and_evaluate(self, 
                           model_class, 
                           model_params: Dict, 
                           noise_std: float, 
                           run_id: int,
                           noise_type: str = "gaussian") -> Tuple[float, float, float, float]:
        
        # Add noise to training data (returns numpy arrays)
        X_train_noisy = self._apply_noise(self.X_train, noise_std, noise_type)
        X_test_noisy  = self._apply_noise(self.X_test,  noise_std, noise_type)
        
        # Create data loaders with noisy data
        train_dataset = TensorDataset(
            torch.tensor(X_train_noisy, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32 if self.binary else torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test_noisy, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.float32 if self.binary else torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.learning_params['batch_size'], shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=self.learning_params['batch_size'], shuffle=False)
        
        if issubclass(model_class, GIFT):
            if 'in_features' not in model_params:
                model_params['in_features'] = self.X_train.shape[1]
            if 'out_features' not in model_params:
                model_params['out_features'] = 1 if self.binary else len(np.unique(self.y_train))
            
            model    = GIFT(**model_params, dtype=torch.float32)
            sk_model = SklearnGIFTWrapper(model, device=self.device)
        else:  # LitANFIS
            if 'in_features' not in model_params:
                model_params['in_features'] = self.X_train.shape[1]
            if 'out_features' not in model_params:
                model_params['out_features'] = 1 if self.binary else len(np.unique(self.y_train))

            model    = LitAnfis(**model_params, dtype=torch.float32)
            sk_model = SklearnLitAnfisWrapper(model, device=self.device)
        
        # Setup training
        optimizer      = torch.optim.Adam(model.parameters(), lr=self.learning_params['lr'])
        steps_per_epoch = len(train_loader)
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_params['max_lr'],
            steps_per_epoch=steps_per_epoch,
            epochs=self.learning_params['epochs']
        )
        
        # Loss functions
        cos = torch.nn.L1Loss()
        if self.binary:
            cross = torch.nn.BCEWithLogitsLoss()
            def criterion(batch_X, batch_y, outputs, reconstructed, alpha):
                return cross(outputs.squeeze(), batch_y.squeeze()) + cos(reconstructed, batch_X) * alpha
        else:
            cross = torch.nn.CrossEntropyLoss()
            def criterion(batch_X, batch_y, outputs, reconstructed, alpha):
                return cross(outputs, batch_y.long()) + cos(reconstructed, batch_X) * alpha
        
        # once per epoch, causing the actual decay to be steps_per_epoch× slower
        # than intended.  Compute a per-epoch multiplier instead.
        alpha     = self.learning_params['alpha']
        min_alpha = self.learning_params['min_alpha']
        decay_epochs = max(self.learning_params['epochs'] / 2, 1)
        alpha_decaying = np.power(min_alpha / alpha, 1.0 / decay_epochs)
        
        model.train()
        for epoch in range(self.learning_params['epochs']):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs, reconstructed, _ = model(batch_X)
                # Reconstruction target is the clean batch_X (intentional)
                loss = criterion(batch_X, batch_y, outputs, reconstructed, alpha)
                loss.backward()
                optimizer.step()
                
                try:
                    scheduler.step()
                except ValueError as e:
                    # OneCycleLR raises ValueError when called beyond total_steps
                    pass
            
            alpha *= alpha_decaying
        
        # Evaluate
        try:
            y_train_pred = sk_model.predict(X_train_noisy)
            y_test_pred  = sk_model.predict(X_test_noisy)
            
            train_acc = accuracy_score(self.y_train, y_train_pred)
            test_acc  = accuracy_score(self.y_test,  y_test_pred)
            
            try:
                train_proba = sk_model.predict_proba(X_train_noisy)
                test_proba  = sk_model.predict_proba(X_test_noisy)
                
                if self.binary:
                    if train_proba.ndim == 1 or train_proba.shape[1] == 1:
                        pos = train_proba.ravel()
                        train_proba = np.stack([1 - pos, pos], axis=1)
                    if test_proba.ndim == 1 or test_proba.shape[1] == 1:
                        pos = test_proba.ravel()
                        test_proba = np.stack([1 - pos, pos], axis=1)

                    train_auc = roc_auc_score(self.y_train, train_proba[:, 1])
                    test_auc  = roc_auc_score(self.y_test,  test_proba[:, 1])
                else:
                    train_auc = roc_auc_score(self.y_train, train_proba, multi_class='ovr')
                    test_auc  = roc_auc_score(self.y_test,  test_proba,  multi_class='ovr')

            except Exception as e:
                print(f"Warning: Could not compute AUC - {e}")
                train_auc, test_auc = 0.0, 0.0
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            train_acc, test_acc, train_auc, test_auc = 0.0, 0.0, 0.0, 0.0
            
        return train_acc, test_acc, train_auc, test_auc
    
    def evaluate(self, verbose: bool = True, noise_type: str = "gaussian") -> pd.DataFrame:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        results_summary = []
        
        for noise_std in tqdm(self.noise_levels, desc="Noise levels"):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Testing noise level: σ = {noise_std}")
                print(f"{'='*50}")
            
            for model_name, model_class, model_params in [
                ('gift',     GIFT,     self.model_params_gift.copy()),
                ('litanfis', LitAnfis, self.model_params_litanfis.copy())
            ]:
                if verbose:
                    print(f"\nTraining {model_name.upper()}...")
                
                # Set the correct dimensions in model_params
                model_params['in_features'] = self.X_train.shape[1]
                if self.binary:
                    model_params['out_features'] = 1
                else:
                    model_params['out_features'] = len(np.unique(self.y_train))
                model_params['binary'] = self.binary
                
                for run in range(self.n_runs):
                    if verbose:
                        print(f"  Run {run + 1}/{self.n_runs}...", end=" ", flush=True)
                    
                    run_seed = self.random_state + run * 100 + int(noise_std * 1000)
                    np.random.seed(run_seed)
                    torch.manual_seed(run_seed)
                    
                    train_acc, test_acc, train_auc, test_auc = self._train_and_evaluate(
                        model_class, model_params, noise_std, run,
                        noise_type=noise_type 
                    )
                    
                    self.results[model_name][noise_std]['train_acc'].append(train_acc)
                    self.results[model_name][noise_std]['test_acc'].append(test_acc)
                    self.results[model_name][noise_std]['train_auc'].append(train_auc)
                    self.results[model_name][noise_std]['test_auc'].append(test_auc)
                    
                    results_summary.append({
                        'model':      model_name,
                        'noise_std':  noise_std,
                        'run':        run,
                        'train_acc':  train_acc,
                        'test_acc':   test_acc,
                        'train_auc':  train_auc,
                        'test_auc':   test_auc
                    })
                    
                    if verbose:
                        print(f"Acc={test_acc:.4f}")
        
        return pd.DataFrame(results_summary)
    
    def get_summary_statistics(self) -> pd.DataFrame:
        summary = []
        
        for model_name in ['gift', 'litanfis']:
            for noise_std in self.noise_levels:
                stats = self.results[model_name][noise_std]
                if len(stats['test_acc']) > 0:
                    summary.append({
                        'model':           model_name,
                        'noise_std':       noise_std,
                        'train_acc_mean':  np.mean(stats['train_acc']),
                        'train_acc_std':   np.std(stats['train_acc']),
                        'test_acc_mean':   np.mean(stats['test_acc']),
                        'test_acc_std':    np.std(stats['test_acc']),
                        'train_auc_mean':  np.mean(stats['train_auc']),
                        'train_auc_std':   np.std(stats['train_auc']),
                        'test_auc_mean':   np.mean(stats['test_auc']),
                        'test_auc_std':    np.std(stats['test_auc'])
                    })
        
        return pd.DataFrame(summary)
    
    def plot_robustness_curves(self, save_path: Optional[str] = None):
        summary_df = self.get_summary_statistics()
        
        if len(summary_df) == 0:
            print("No data to plot. Run evaluate() first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1 = axes[0]
        for model_name in ['gift', 'litanfis']:
            model_data = summary_df[summary_df['model'] == model_name]
            if len(model_data) > 0:
                x      = model_data['noise_std'].values
                y_mean = model_data['test_acc_mean'].values
                y_std  = model_data['test_acc_std'].values
                
                ax1.plot(x, y_mean, 'o-', label=model_name.upper(), linewidth=2, markersize=8)
                ax1.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        
        ax1.set_xlabel('Noise Standard Deviation (σ)', fontsize=12)
        ax1.set_ylabel('Test Accuracy', fontsize=12)
        ax1.set_title('Robustness Comparison: Test Accuracy', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        for model_name in ['gift', 'litanfis']:
            model_data = summary_df[summary_df['model'] == model_name]
            if len(model_data) > 0:
                x      = model_data['noise_std'].values
                y_mean = model_data['test_auc_mean'].values
                y_std  = model_data['test_auc_std'].values
                
                ax2.plot(x, y_mean, 'o-', label=model_name.upper(), linewidth=2, markersize=8)
                ax2.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        
        ax2.set_xlabel('Noise Standard Deviation (σ)', fontsize=12)
        ax2.set_ylabel('Test AUC', fontsize=12)
        ax2.set_title('Robustness Comparison: Test AUC', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_relative_performance(self, save_path: Optional[str] = None):
        summary_df = self.get_summary_statistics()
        
        if len(summary_df) == 0:
            print("No data to plot. Run evaluate() first.")
            return
        
        gift_data     = summary_df[summary_df['model'] == 'gift']
        litanfis_data = summary_df[summary_df['model'] == 'litanfis']
        
        if len(gift_data) == 0 or len(litanfis_data) == 0:
            print("Missing data for one of the models.")
            return
        
        x               = gift_data['noise_std'].values
        acc_improvement = gift_data['test_acc_mean'].values - litanfis_data['test_acc_mean'].values
        auc_improvement = gift_data['test_auc_mean'].values - litanfis_data['test_auc_mean'].values
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1    = axes[0]
        colors = ['green' if v > 0 else 'red' for v in acc_improvement]
        bars1  = ax1.bar(range(len(x)), acc_improvement, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xticks(range(len(x)))
        ax1.set_xticklabels([f'{v:.2f}' for v in x])
        ax1.set_xlabel('Noise Standard Deviation (σ)', fontsize=12)
        ax1.set_ylabel('Accuracy Improvement (GIFT - LitANFIS)', fontsize=12)
        ax1.set_title('Relative Test Accuracy', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, acc_improvement):
            height = bar.get_height()
            offset = 0.01 * max(abs(acc_improvement)) if len(acc_improvement) > 0 else 0.01
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height + offset if val >= 0 else height - offset,
                f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9
            )
        
        ax2    = axes[1]
        colors = ['green' if v > 0 else 'red' for v in auc_improvement]
        bars2  = ax2.bar(range(len(x)), auc_improvement, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xticks(range(len(x)))
        ax2.set_xticklabels([f'{v:.2f}' for v in x])
        ax2.set_xlabel('Noise Standard Deviation (σ)', fontsize=12)
        ax2.set_ylabel('AUC Improvement (GIFT - LitANFIS)', fontsize=12)
        ax2.set_title('Relative Test AUC', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, auc_improvement):
            height = bar.get_height()
            offset = 0.01 * max(abs(auc_improvement)) if len(auc_improvement) > 0 else 0.01
            ax2.text(
                bar.get_x() + bar.get_width() / 2.,
                height + offset if val >= 0 else height - offset,
                f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compute_robustness_score(self) -> Dict:
        summary_df = self.get_summary_statistics()
        
        if len(summary_df) == 0:
            return {}
        
        robustness_scores = {}
        
        for model_name in ['gift', 'litanfis']:
            model_data = summary_df[summary_df['model'] == model_name].sort_values('noise_std')
            
            if len(model_data) == 0:
                continue
                
            x     = model_data['noise_std'].values
            y_acc = model_data['test_acc_mean'].values
            y_auc = model_data['test_auc_mean'].values
            
            baseline_acc = y_acc[0] if len(y_acc) > 0 else 0
            baseline_auc = y_auc[0] if len(y_auc) > 0 else 0
            
            y_acc_norm = y_acc / baseline_acc if baseline_acc > 0 else y_acc
            y_auc_norm = y_auc / baseline_auc if baseline_auc > 0 else y_auc
            
            acc_auc = np.trapz(y_acc_norm, x)
            auc_auc = np.trapz(y_auc_norm, x)
            
            robustness_scores[model_name] = {
                'accuracy_robustness_score': acc_auc,
                'auc_robustness_score':      auc_auc,
                'combined_score':            (acc_auc + auc_auc) / 2,
                'baseline_acc':              baseline_acc,
                'baseline_auc':              baseline_auc
            }
        
        return robustness_scores
    
    def get_best_model(self) -> Tuple[str, Dict]:
        robustness_scores = self.compute_robustness_score()
        summary_df        = self.get_summary_statistics()
        
        if len(robustness_scores) < 2:
            return "Insufficient data", {}
        
        criteria = {
            'accuracy_robustness': robustness_scores['gift']['accuracy_robustness_score'] > robustness_scores['litanfis']['accuracy_robustness_score'],
            'auc_robustness':      robustness_scores['gift']['auc_robustness_score']      > robustness_scores['litanfis']['auc_robustness_score'],
            'combined_score':      robustness_scores['gift']['combined_score']             > robustness_scores['litanfis']['combined_score']
        }
        
        max_noise           = max(self.noise_levels)
        gift_high_noise     = summary_df[(summary_df['model'] == 'gift')     & (summary_df['noise_std'] == max_noise)]['test_acc_mean'].values
        litanfis_high_noise = summary_df[(summary_df['model'] == 'litanfis') & (summary_df['noise_std'] == max_noise)]['test_acc_mean'].values
        
        if len(gift_high_noise) > 0 and len(litanfis_high_noise) > 0:
            criteria['high_noise_accuracy'] = gift_high_noise[0] > litanfis_high_noise[0]
        else:
            criteria['high_noise_accuracy'] = False
        
        gift_wins     = sum(criteria.values())
        litanfis_wins = len(criteria) - gift_wins
        
        if gift_wins > litanfis_wins:
            best_model = 'GIFT'
            reason     = f"GIFT wins on {gift_wins}/{len(criteria)} criteria"
        elif litanfis_wins > gift_wins:
            best_model = 'LitANFIS'
            reason     = f"LitANFIS wins on {litanfis_wins}/{len(criteria)} criteria"
        else:
            if robustness_scores['gift']['combined_score'] >= robustness_scores['litanfis']['combined_score']:
                best_model = 'GIFT'
            else:
                best_model = 'LitANFIS'
            reason = "Tie broken by combined robustness score"
        
        return best_model, {
            'criteria':          criteria,
            'robustness_scores': robustness_scores,
            'decision_reason':   reason,
            'gift_wins':         gift_wins,
            'litanfis_wins':     litanfis_wins
        }
    
    def print_detailed_report(self):
        print("\n" + "="*80)
        print("ROBUSTNESS EVALUATION REPORT")
        print(f"Dataset: {self.experiment.__class__.__name__}")
        print(f"Number of runs per noise level: {self.n_runs}")
        print("="*80)
        
        summary_df = self.get_summary_statistics()
        if len(summary_df) > 0:
            print("\n📊 PERFORMANCE SUMMARY:")
            print("-"*80)
            print(f"{'Model':<12} {'Noise σ':<8} {'Test Acc ± Std':<20} {'Test AUC ± Std':<20}")
            print("-"*80)
            
            for model_name in ['gift', 'litanfis']:
                model_data = summary_df[summary_df['model'] == model_name]
                for _, row in model_data.iterrows():
                    print(f"{model_name.upper():<12} {row['noise_std']:<8.2f} "
                          f"{row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f}    "
                          f"{row['test_auc_mean']:.4f} ± {row['test_auc_std']:.4f}")
        else:
            print("\n⚠️ No results available. Run evaluate() first.")
            return
        
        robustness_scores = self.compute_robustness_score()
        if robustness_scores:
            print("\n🎯 ROBUSTNESS SCORES (Area under performance curve):")
            print("-"*80)
            for model_name in ['gift', 'litanfis']:
                if model_name in robustness_scores:
                    scores = robustness_scores[model_name]
                    print(f"{model_name.upper()}:")
                    print(f"  • Accuracy Robustness: {scores['accuracy_robustness_score']:.4f}")
                    print(f"  • AUC Robustness:      {scores['auc_robustness_score']:.4f}")
                    print(f"  • Combined Score:      {scores['combined_score']:.4f}")
                    print(f"  • Baseline (σ=0):      Acc={scores['baseline_acc']:.4f}, AUC={scores['baseline_auc']:.4f}")
        
        best_model, details = self.get_best_model()
        print("\n🏆 ROBUSTNESS VERDICT:")
        print("-"*80)
        print(f"✅ MOST ROBUST MODEL: {best_model}")
        if 'decision_reason' in details:
            print(f"📝 Reason: {details['decision_reason']}")
        
        if 'criteria' in details:
            print(f"\nDetailed criteria comparison:")
            for criterion, gift_won in details['criteria'].items():
                winner = "GIFT" if gift_won else "LitANFIS"
                print(f"  • {criterion}: {winner}")
        
        if 'gift_wins' in details:
            print(f"\nFinal count: GIFT={details['gift_wins']}, LitANFIS={details['litanfis_wins']}")
        print("="*80)