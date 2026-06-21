import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional, Type, Union, Any
from torch.optim.lr_scheduler import OneCycleLR
import warnings
import random

warnings.filterwarnings('ignore')


class Evaluator:
    """
    Flexible robustness evaluator that can compare any two models with optional noise
    """
    
    def __init__(self, 
                 experiment,
                 model_configs: Dict[str, Dict], 
                 learning_params: Dict,
                 device: torch.device,
                 binary: bool = True,
                 noise_levels: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                 n_runs: int = 3,
                 random_state: int = 42,
                 use_noise: bool = True,  
                 noise_type: str = "gaussian"):

        self.experiment = experiment
        self.model_configs = model_configs 
        self.learning_params = learning_params
        self.device = device
        self.binary = binary
        self.noise_levels = noise_levels if use_noise else [0.0]
        self.n_runs = n_runs
        self.random_state = random_state
        self.use_noise = use_noise 
        self.noise_type = noise_type
        
        # Validate noise type
        valid_noise_types = ["gaussian", "salt_pepper"]
        if noise_type not in valid_noise_types:
            raise ValueError(f"noise_type must be one of {valid_noise_types}, got '{noise_type}'")
        
        # Dynamically create results storage for all models
        self.results = {}
        for model_name in model_configs.keys():
            self.results[model_name] = {
                nl: {'train_acc': [], 'test_acc': [], 'train_auc': [], 'test_auc': [], 'linguistic_richness': []}
                for nl in self.noise_levels
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
        """Apply noise to data if use_noise is True and noise_std > 0"""
        
        if not self.use_noise or noise_std == 0:
            return X
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        if noise_type == "gaussian":
            noise = torch.randn_like(X_tensor) * noise_std
            result = X_tensor + noise
            
        elif noise_type == "salt_pepper":
            salt_vs_pepper_ratio = 0.5
            random_mask = torch.rand_like(X_tensor)
            
            salt_mask = random_mask < (noise_std * salt_vs_pepper_ratio)
            pepper_mask = (random_mask >= (noise_std * salt_vs_pepper_ratio)) & (random_mask < noise_std)
            
            result = X_tensor.clone()
            
            if len(X.shape) == 2:
                max_val = torch.max(X_tensor)
                min_val = torch.min(X_tensor)
                result[salt_mask] = max_val
                result[pepper_mask] = min_val
            else:
                result[salt_mask] = 1.0 if torch.max(X_tensor) <= 1.0 else torch.max(X_tensor)
                result[pepper_mask] = 0.0 if torch.min(X_tensor) >= 0.0 else torch.min(X_tensor)
        else:
            raise ValueError(
                f"Unknown noise type: {noise_type}. "
                "Choose from: 'gaussian' or 'salt_pepper'"
            )
        
        return result.numpy()
    
    @staticmethod
    def _safe_multiclass_auc(y_true: np.ndarray, y_proba: np.ndarray, all_classes: np.ndarray) -> float:
        """
        One-vs-rest macro AUC that skips classes absent from y_true instead of
        letting sklearn's undefined per-class score (NaN) silently poison the
        averaged result. Matches roc_auc_score(..., multi_class='ovr') exactly
        when every class in `all_classes` is present in y_true.

        y_proba columns are assumed to be ordered to match `all_classes`
        (this holds for predict_proba output paired with np.unique-derived labels).
        
        Returns NaN only if NO classes can be evaluated (e.g., test set is single-class
        or perfectly predicted). In edge cases with perfect predictions, uses accuracy-based
        fallback.
        """
        aucs = []
        for i, cls in enumerate(all_classes):
            y_bin = (y_true == cls).astype(int)
            n_pos = y_bin.sum()
            if n_pos == 0 or n_pos == len(y_bin):
                # Class missing from this split (or split is single-class):
                # OvR AUC is undefined here, so skip rather than inject NaN.
                continue
            aucs.append(roc_auc_score(y_bin, y_proba[:, i]))
        
        if aucs:
            return float(np.mean(aucs))
        else:
            # Edge case: test set is single-class or all classes perfectly separated
            # Use 1.0 as AUC if predictions are perfect (perfect accuracy case)
            predictions = np.argmax(y_proba, axis=1)
            if np.array_equal(predictions, y_true):
                # All predictions correct → AUC would be 1.0 if computable
                return 1.0
            else:
                # Unable to compute meaningful AUC
                return np.nan

    def _train_and_evaluate(self, 
                           model_class: Type[nn.Module],
                           model_params: Dict,
                           wrapper_class: Type,
                           noise_std: float, 
                           run_id: int,
                           run_seed: int = 42) -> Tuple[float, float, float, float, float, Any]:
        """
        Train and evaluate a single model with given noise level
        Returns: (train_acc, test_acc, train_auc, test_auc, linguistic_richness, trained_wrapper)

        linguistic_richness is the mean per-rule entropy (in nats), computed
        on the SAME shared 4-category support (equal / not-equal /
        greater-than / less-than — see linguistic_richness_utils.py) for
        every model class that exposes `model.linguistic_richness()`
        (ANFIS, UNFIS, GRIFFIN, LitAnfis, GIFTSHIFT, GIFTSHIFTER all do).
        Because the support is shared, the value is directly comparable
        across model classes without renormalization:
          - ANFIS / UNFIS have no relational gate at all -> structurally 0.0
          - LitAnfis / GRIFFIN have only an equal/not-equal gate ->
            structurally capped at log(2) ~= 0.693
          - GIFTSHIFT / GIFTSHIFTER have both branches -> can reach log(4)
        It is np.nan only for model classes that don't define the method
        at all, or when training diverged to NaN/Inf (see the guard below),
        since 0.0 there would misleadingly read as "rich but degenerate"
        rather than "not applicable" / "failed run".
        """
        # Add noise to data
        X_train_noisy = self._apply_noise(self.X_train, noise_std, self.noise_type)
        X_test_noisy = self._apply_noise(self.X_test, noise_std, self.noise_type)
        
        # Create data loaders
        if self.binary:
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        else:
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
            
        train_dataset = TensorDataset(
            torch.tensor(X_train_noisy, dtype=torch.float32),
            y_train_tensor
        )
        
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(run_seed if self.random_state is not None else torch.initial_seed())

        train_loader = DataLoader(train_dataset, batch_size=self.learning_params['batch_size'], shuffle=True,
                                  worker_init_fn=seed_worker, generator=g)
        
        # Initialize model
        model = model_class(**model_params, dtype=torch.float32)
        model = model.to(self.device)
        
        # Check if model has entropy_coef attribute (for GIFTSHIFTENTROPY)
        has_entropy_reg = hasattr(model, 'entropy_coef')
        entropy_coef = model.entropy_coef if has_entropy_reg else 0.0
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_params['lr'])
        steps_per_epoch = len(train_loader)
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_params['max_lr'],
            steps_per_epoch=steps_per_epoch,
            epochs=self.learning_params['epochs']
        )
        
        # Loss functions
        cos = torch.nn.L1Loss()
        
        # Define criterion that handles both model types
        if self.binary:
            cross = torch.nn.BCEWithLogitsLoss()
            def criterion(batch_X, batch_y, outputs, reconstructed, alpha, entropy_penalty=None):
                main_loss = cross(outputs.squeeze(), batch_y.squeeze())
                recon_loss = cos(reconstructed, batch_X) * alpha
                if entropy_penalty is not None and has_entropy_reg:
                    # entropy_penalty is a [B, R] tensor — reduce to scalar before scaling
                    entropy_loss = entropy_penalty.mean() * entropy_coef
                    return main_loss + recon_loss + entropy_loss
                return main_loss + recon_loss
        else:
            cross = torch.nn.CrossEntropyLoss()
            def criterion(batch_X, batch_y, outputs, reconstructed, alpha, entropy_penalty=None):
                main_loss = cross(outputs, batch_y.long())
                recon_loss = cos(reconstructed, batch_X) * alpha
                if entropy_penalty is not None and has_entropy_reg:
                    # entropy_penalty is a [B, R] tensor — reduce to scalar before scaling
                    entropy_loss = entropy_penalty.mean() * entropy_coef
                    return main_loss + recon_loss + entropy_loss
                return main_loss + recon_loss
        
        # Alpha decay
        # FIX Bug 3: guard against min_alpha=0 which makes power(0/alpha) collapse to 0 instantly
        alpha = self.learning_params['alpha']
        min_alpha = self.learning_params['min_alpha']
        safe_min_alpha = max(min_alpha, 1e-6)
        decay_epochs = max(self.learning_params['epochs'] / 2, 1)
        
        if alpha > 0:
            alpha_decaying = np.power(safe_min_alpha / alpha, 1.0 / (steps_per_epoch * decay_epochs))
        else:
            alpha_decaying = 0.95
        
        # Training loop
        for epoch in range(self.learning_params['epochs']):
            model.train()
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass - handles both GIFTSHIFT and GIFTSHIFTENTROPY
                outputs = model(batch_X)
                
                # Unpack based on return length
                if len(outputs) == 3:
                    preds, reconstructed, entropy_penalty = outputs
                else:
                    preds, reconstructed = outputs
                    entropy_penalty = None
                
                loss = criterion(batch_X, batch_y, preds, reconstructed, alpha, entropy_penalty)
                loss.backward()
                optimizer.step()
                
                try:
                    scheduler.step()
                except ValueError:
                    pass
            
            alpha = max(safe_min_alpha, alpha * alpha_decaying)
        
        # Switch to evaluation mode
        model.eval()
        
        # Guard against silent NaN collapse: if training diverged (e.g. a
        # parameter underflowed/blew up), predict()/predict_proba() will
        # still "succeed" but produce NaN-derived, effectively-random output
        # that gets silently accepted as a real data point below. Catch it
        # here instead of letting it pollute the averaged results.
        if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
            print(f"Warning: model diverged to NaN/Inf (run {run_id}, noise_std={noise_std}) — recording as failed run")
            wrapper = wrapper_class(model, device=self.device)
            return 0.0, 0.0, 0.0, 0.0, np.nan, wrapper
        
        # Create wrapper with trained model
        wrapper = wrapper_class(model, device=self.device)
        
        # Linguistic richness: every model class in this codebase now
        # exposes model.linguistic_richness() on the shared 4-category
        # scale (ANFIS/UNFIS report a structural 0.0; LitAnfis/GRIFFIN are
        # capped at log(2); GIFTSHIFT/GIFTSHIFTER can reach log(4)). NaN is
        # reserved for model classes that genuinely don't define the
        # method, or if computing it raises for some other reason.
        if hasattr(model, 'linguistic_richness'):
            try:
                linguistic_richness = model.linguistic_richness()
            except Exception as e:
                print(f"Warning: Could not compute linguistic_richness - {e}")
                linguistic_richness = np.nan
        else:
            linguistic_richness = np.nan
        
        # Evaluation
        try:
            y_train_pred = wrapper.predict(X_train_noisy)
            y_test_pred = wrapper.predict(X_test_noisy)
            
            train_acc = accuracy_score(self.y_train, y_train_pred)
            test_acc = accuracy_score(self.y_test, y_test_pred)
            
            try:
                train_proba = wrapper.predict_proba(X_train_noisy)
                test_proba = wrapper.predict_proba(X_test_noisy)
                
                if self.binary:
                    # FIX Bug 4: guard ndim before accessing shape[1] to avoid IndexError on 1D arrays
                    if train_proba.ndim == 1 or (train_proba.ndim == 2 and train_proba.shape[1] == 1):
                        pos = train_proba.ravel()
                        train_proba = np.stack([1 - pos, pos], axis=1)
                    if test_proba.ndim == 1 or (test_proba.ndim == 2 and test_proba.shape[1] == 1):
                        pos = test_proba.ravel()
                        test_proba = np.stack([1 - pos, pos], axis=1)
                    
                    train_auc = roc_auc_score(self.y_train, train_proba[:, 1])
                    test_auc = roc_auc_score(self.y_test, test_proba[:, 1])
                else:
                    # For multiclass, average per-class OvR AUC only over classes
                    # that actually appear in this split's y_true.
                    all_classes = np.unique(np.concatenate([self.y_train, self.y_test]))
                    #
                    # FIX: the original code passed labels=union(train, test) to
                    # roc_auc_score(..., multi_class='ovr') for BOTH train and test.
                    # With many classes and few samples per class (e.g. ORL: 40
                    # classes, ~5-10 samples each), it's common for a class present
                    # in train to be absent from test (or vice versa). sklearn
                    # requires `labels` to have the same length as the probability
                    # matrix's columns, so labels can't simply be narrowed to the
                    # present classes while keeping all proba columns — and OvR AUC
                    # is mathematically undefined for a class with zero positives.
                    # Passing it anyway produces a per-class NaN that silently
                    # propagates into the averaged score. No exception is raised
                    # (this file suppresses warnings globally), so it isn't caught
                    # by the except block below, and you silently get test_auc=nan.
                    #
                    # Fix: compute each class's OvR AUC individually and average
                    # only over classes with both positive and negative examples
                    # in that split, skipping (not zeroing) the rest.
                    train_auc = self._safe_multiclass_auc(self.y_train, train_proba, all_classes)
                    test_auc = self._safe_multiclass_auc(self.y_test, test_proba, all_classes)
                    
            except Exception as e:
                print(f"Warning: Could not compute AUC - {e}")
                train_auc, test_auc = 0.0, 0.0
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            train_acc, test_acc, train_auc, test_auc = 0.0, 0.0, 0.0, 0.0
            
        return train_acc, test_acc, train_auc, test_auc, linguistic_richness, wrapper
    
    def evaluate(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run the full evaluation for all models
        """
        results_summary = []
        
        # Display noise status
        if not self.use_noise:
            print("\n" + "="*60)
            print("🔇 NOISE IS DISABLED - Clean data evaluation only")
            print("="*60)
        else:
            print("\n" + "="*60)
            print(f"🔊 NOISE IS ENABLED - Type: {self.noise_type}")
            print("="*60)
        
        for noise_std in tqdm(self.noise_levels, desc="Noise levels"):
            if verbose:
                print(f"\n{'='*50}")
                if noise_std == 0:
                    print(f"Testing: CLEAN DATA (σ = 0)")
                else:
                    print(f"Testing noise level: σ = {noise_std}")
                print(f"{'='*50}")
            
            for model_name, config in self.model_configs.items():
                if verbose:
                    print(f"\nTraining {model_name.upper()}...")
                
                # Prepare model parameters
                # Accept either 'model_params' or 'params' as the key
                raw_params = config.get('model_params', config.get('params', {}))
                model_params = raw_params.copy()
                # Only set defaults for in_features / out_features / binary
                # if the caller did NOT already supply them (e.g. via PCA wrapper)
                if 'in_features' not in model_params:
                    model_params['in_features'] = self.X_train.shape[1]
                if 'out_features' not in model_params:
                    if self.binary:
                        model_params['out_features'] = 1
                    else:
                        model_params['out_features'] = len(np.unique(self.y_train))
                if 'binary' not in model_params:
                    model_params['binary'] = self.binary
                
                for run in range(self.n_runs):
                    if verbose:
                        print(f"  Run {run + 1}/{self.n_runs}...", end=" ", flush=True)
                    
                    # Set seed for this run
                    if self.random_state is not None:
                        run_seed = self.random_state + run * 100 + int(noise_std * 1000)
                    else:
                        run_seed = np.random.randint(0, 2**32 - 1)
                    
                    np.random.seed(run_seed)
                    random.seed(run_seed)
                    torch.manual_seed(run_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(run_seed)
                    
                    train_acc, test_acc, train_auc, test_auc, linguistic_richness, wrapper = self._train_and_evaluate(
                        config['model_class'],
                        model_params,
                        config['wrapper_class'],
                        noise_std,
                        run,
                        run_seed
                    )
                    
                    if verbose:
                        lr_str = f", LingRich={linguistic_richness:.4f}" if not np.isnan(linguistic_richness) else ""
                        print(f"Acc={test_acc:.4f}{lr_str}")
                    
                    self.results[model_name][noise_std]['train_acc'].append(train_acc)
                    self.results[model_name][noise_std]['test_acc'].append(test_acc)
                    self.results[model_name][noise_std]['train_auc'].append(train_auc)
                    self.results[model_name][noise_std]['test_auc'].append(test_auc)
                    self.results[model_name][noise_std]['linguistic_richness'].append(linguistic_richness)
                    
                    results_summary.append({
                        'model': model_name,
                        'noise_std': noise_std,
                        'run': run,
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'train_auc': train_auc,
                        'test_auc': test_auc,
                        'linguistic_richness': linguistic_richness
                    })
        
        return pd.DataFrame(results_summary)
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all models"""
        summary = []
        
        for model_name in self.model_configs.keys():
            for noise_std in self.noise_levels:
                stats = self.results[model_name][noise_std]
                if len(stats['test_acc']) > 0:
                    # linguistic_richness is NaN for model classes that don't
                    # define it (e.g. plain ANFIS/LitAnfis/GRIFFIN). Use
                    # nan-safe aggregation so a non-applicable model doesn't
                    # turn the whole column into NaN; if EVERY run is NaN
                    # (metric truly not applicable for this model), keep NaN.
                    lr_values = stats['linguistic_richness']
                    if len(lr_values) > 0 and not all(np.isnan(v) for v in lr_values):
                        lr_mean = np.nanmean(lr_values)
                        lr_std = np.nanstd(lr_values)
                    else:
                        lr_mean, lr_std = np.nan, np.nan
                    
                    summary.append({
                        'model': model_name,
                        'noise_std': noise_std,
                        'train_acc_mean': np.mean(stats['train_acc']),
                        'train_acc_std': np.std(stats['train_acc']),
                        'test_acc_mean': np.mean(stats['test_acc']),
                        'test_acc_std': np.std(stats['test_acc']),
                        'train_auc_mean': np.mean(stats['train_auc']),
                        'train_auc_std': np.std(stats['train_auc']),
                        'test_auc_mean': np.mean(stats['test_auc']),
                        'test_auc_std': np.std(stats['test_auc']),
                        'linguistic_richness_mean': lr_mean,
                        'linguistic_richness_std': lr_std
                    })
        
        return pd.DataFrame(summary)
    
    def plot_robustness_curves(self, save_path: Optional[str] = None):
        """Plot robustness curves for all models"""
        summary_df = self.get_summary_statistics()
        
        if len(summary_df) == 0:
            print("No data to plot. Run evaluate() first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot Accuracy
        ax1 = axes[0]
        for model_name in self.model_configs.keys():
            model_data = summary_df[summary_df['model'] == model_name]
            if len(model_data) > 0:
                x = model_data['noise_std'].values
                y_mean = model_data['test_acc_mean'].values
                y_std = model_data['test_acc_std'].values
                
                ax1.plot(x, y_mean, 'o-', label=model_name.upper(), linewidth=2, markersize=8)
                ax1.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        
        ax1.set_xlabel('Noise Standard Deviation (σ)' if self.use_noise else 'Experiment', fontsize=12)
        ax1.set_ylabel('Test Accuracy', fontsize=12)
        ax1.set_title('Model Comparison: Test Accuracy', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot AUC
        ax2 = axes[1]
        for model_name in self.model_configs.keys():
            model_data = summary_df[summary_df['model'] == model_name]
            if len(model_data) > 0:
                x = model_data['noise_std'].values
                y_mean = model_data['test_auc_mean'].values
                y_std = model_data['test_auc_std'].values
                
                ax2.plot(x, y_mean, 'o-', label=model_name.upper(), linewidth=2, markersize=8)
                ax2.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        
        ax2.set_xlabel('Noise Standard Deviation (σ)' if self.use_noise else 'Experiment', fontsize=12)
        ax2.set_ylabel('Test AUC', fontsize=12)
        ax2.set_title('Model Comparison: Test AUC', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_relative_performance(self, save_path: Optional[str] = None):
        """Plot relative performance between models"""
        summary_df = self.get_summary_statistics()
        
        if len(summary_df) == 0:
            print("No data to plot. Run evaluate() first.")
            return
        
        model_names = list(self.model_configs.keys())
        if len(model_names) != 2:
            print("Relative performance plot requires exactly 2 models")
            return
        
        model1_data = summary_df[summary_df['model'] == model_names[0]]
        model2_data = summary_df[summary_df['model'] == model_names[1]]
        
        if len(model1_data) == 0 or len(model2_data) == 0:
            print("Missing data for one of the models.")
            return
        
        x = model1_data['noise_std'].values
        acc_improvement = model1_data['test_acc_mean'].values - model2_data['test_acc_mean'].values
        auc_improvement = model1_data['test_auc_mean'].values - model2_data['test_auc_mean'].values
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy improvement
        ax1 = axes[0]
        colors = ['green' if v > 0 else 'red' for v in acc_improvement]
        bars1 = ax1.bar(range(len(x)), acc_improvement, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xticks(range(len(x)))
        ax1.set_xticklabels([f'{v:.2f}' for v in x])
        ax1.set_xlabel('Noise Standard Deviation (σ)' if self.use_noise else 'Experiment', fontsize=12)
        ax1.set_ylabel(f'Accuracy Improvement ({model_names[0].upper()} - {model_names[1].upper()})', fontsize=12)
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
        
        # AUC improvement
        ax2 = axes[1]
        colors = ['green' if v > 0 else 'red' for v in auc_improvement]
        bars2 = ax2.bar(range(len(x)), auc_improvement, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xticks(range(len(x)))
        ax2.set_xticklabels([f'{v:.2f}' for v in x])
        ax2.set_xlabel('Noise Standard Deviation (σ)' if self.use_noise else 'Experiment', fontsize=12)
        ax2.set_ylabel(f'AUC Improvement ({model_names[0].upper()} - {model_names[1].upper()})', fontsize=12)
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
        """Compute robustness scores for all models"""
        summary_df = self.get_summary_statistics()
        
        if len(summary_df) == 0:
            return {}
        
        robustness_scores = {}
        
        for model_name in self.model_configs.keys():
            model_data = summary_df[summary_df['model'] == model_name].sort_values('noise_std')
            
            if len(model_data) == 0:
                continue
            
            x = model_data['noise_std'].values
            y_acc = model_data['test_acc_mean'].values
            y_auc = model_data['test_auc_mean'].values
            
            baseline_acc = y_acc[0] if len(y_acc) > 0 else 0
            baseline_auc = y_auc[0] if len(y_auc) > 0 else 0
            
            y_acc_norm = y_acc / baseline_acc if baseline_acc > 0 else y_acc
            y_auc_norm = y_auc / baseline_auc if baseline_auc > 0 else y_auc
            
            if self.use_noise and len(x) > 1:
                acc_auc = np.trapz(y_acc_norm, x)
                auc_auc = np.trapz(y_auc_norm, x)
            else:
                acc_auc = y_acc_norm[0] if len(y_acc_norm) > 0 else 0
                auc_auc = y_auc_norm[0] if len(y_auc_norm) > 0 else 0
            
            robustness_scores[model_name] = {
                'accuracy_robustness_score': acc_auc,
                'auc_robustness_score': auc_auc,
                'combined_score': (acc_auc + auc_auc) / 2,
                'baseline_acc': baseline_acc,
                'baseline_auc': baseline_auc
            }
        
        return robustness_scores
    
    def get_best_model(self) -> Tuple[str, Dict]:
        """Determine the best model based on robustness scores (if noise enabled) or standard performance (if clean data only)"""
        
        summary_df = self.get_summary_statistics()
        
        if len(summary_df) == 0:
            return "Insufficient data", {}
        
        model_names = list(self.model_configs.keys())
        if len(model_names) < 2:
            return "Insufficient data", {}

        use_robustness = self.use_noise and len(self.noise_levels) > 1

        if use_robustness:
            robustness_scores = self.compute_robustness_score()
            if len(robustness_scores) < 2:
                print("Warning: Robustness scores not available, falling back to standard metrics")
                use_robustness = False

        if use_robustness:
            # ROBUSTNESS-BASED COMPARISON — works for N >= 2 models
            # Score each model by its combined robustness score
            scored = {
                name: robustness_scores[name]['combined_score']
                for name in model_names
                if name in robustness_scores
            }
            if not scored:
                return "Insufficient data", {}

            best_model = max(scored, key=scored.get)
            reason = (
                f"{best_model.upper()} has the highest combined robustness score "
                f"({scored[best_model]:.4f}) across {len(model_names)} models"
            )
            return best_model.upper(), {
                'robustness_scores': robustness_scores,
                'all_combined_scores': {k: v for k, v in scored.items()},
                'decision_reason': reason,
                'decision_type': 'robustness_based',
            }

        else:
            # STANDARD PERFORMANCE-BASED COMPARISON (Clean data) — works for N >= 2 models
            clean_data = summary_df[summary_df['noise_std'] == 0]
            if len(clean_data) == 0:
                clean_data = summary_df[summary_df['noise_std'] == summary_df['noise_std'].min()]

            clean_performance = {}
            for name in model_names:
                row = clean_data[clean_data['model'] == name]
                if len(row) == 0:
                    continue
                clean_performance[name] = {
                    'accuracy':     row['test_acc_mean'].values[0],
                    'accuracy_std': row['test_acc_std'].values[0],
                    'auc':          row['test_auc_mean'].values[0],
                    'auc_std':      row['test_auc_std'].values[0],
                    'combined':     (row['test_acc_mean'].values[0] + row['test_auc_mean'].values[0]) / 2,
                }

            if not clean_performance:
                return "Insufficient data", {}

            best_model = max(clean_performance, key=lambda k: clean_performance[k]['combined'])
            best = clean_performance[best_model]
            reason = (
                f"{best_model.upper()} has the highest combined accuracy+AUC on clean data "
                f"(Acc={best['accuracy']:.4f}, AUC={best['auc']:.4f})"
            )
            return best_model.upper(), {
                'clean_performance': clean_performance,
                'decision_reason': reason,
                'decision_type': 'performance_based',
            }
    
    def print_detailed_report(self):
        """Print a detailed report of the evaluation"""
        print("\n" + "="*80)
        print("MODEL EVALUATION REPORT")
        print(f"Dataset: {self.experiment.__class__.__name__}")
        print(f"Noise Enabled: {self.use_noise}")
        if self.use_noise:
            print(f"Noise Type: {self.noise_type}")
        print(f"Number of runs per configuration: {self.n_runs}")
        print("="*80)
        
        summary_df = self.get_summary_statistics()
        if len(summary_df) > 0:
            print("\n📊 PERFORMANCE SUMMARY:")
            print("-"*98)
            print(f"{'Model':<15} {'Noise σ':<10} {'Test Acc ± Std':<20} {'Test AUC ± Std':<20} {'Ling. Richness ± Std':<22}")
            print("-"*98)
            
            for model_name in self.model_configs.keys():
                model_data = summary_df[summary_df['model'] == model_name]
                for _, row in model_data.iterrows():
                    noise_label = f"{row['noise_std']:.2f}" if self.use_noise else "Clean"
                    if np.isnan(row['linguistic_richness_mean']):
                        lr_label = "N/A"
                    else:
                        lr_label = f"{row['linguistic_richness_mean']:.4f} ± {row['linguistic_richness_std']:.4f}"
                    print(f"{model_name.upper():<15} {noise_label:<10} "
                          f"{row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f}    "
                          f"{row['test_auc_mean']:.4f} ± {row['test_auc_std']:.4f}    "
                          f"{lr_label:<22}")
        else:
            print("\n⚠️ No results available. Run evaluate() first.")
            return
        
        robustness_scores = self.compute_robustness_score()
        if robustness_scores and self.use_noise and len(self.noise_levels) > 1:
            print("\n🎯 ROBUSTNESS SCORES:")
            print("-"*80)
            for model_name in self.model_configs.keys():
                if model_name in robustness_scores:
                    scores = robustness_scores[model_name]
                    print(f"{model_name.upper()}:")
                    print(f"  • Accuracy Score: {scores['accuracy_robustness_score']:.4f}")
                    print(f"  • AUC Score:      {scores['auc_robustness_score']:.4f}")
                    print(f"  • Combined Score: {scores['combined_score']:.4f}")
                    print(f"  • Baseline (clean): Acc={scores['baseline_acc']:.4f}, AUC={scores['baseline_auc']:.4f}")
        
        # Linguistic richness: descriptive only (not folded into best-model
        # selection), shown on the clean-data (noise_std == 0) split for the
        # subset of models that expose model.linguistic_richness().
        #
        # As of this version, EVERY model class computes its richness on
        # the SAME shared 4-category support (equal / not-equal /
        # greater-than / less-than — see linguistic_richness_utils.py),
        # so the raw nats value H is now directly comparable across model
        # classes with no per-model renormalization needed. A model with
        # no relational (greater/less) branch (LitAnfis, GRIFFIN) is
        # structurally capped at log(2) ≈ 0.693; a model with no
        # relational gate at all (ANFIS, UNFIS) is structurally fixed at
        # H = 0.0. Both are real, comparable consequences of the
        # architecture, not scale artifacts — that's the point of using
        # a shared support instead of grading each model on its own curve.
        #
        # The shared ceiling log(4) ≈ 1.386 is the same for every model
        # and is shown once at the top of the report rather than per row.
        clean_lr = summary_df[summary_df['noise_std'] == 0] if self.use_noise else summary_df
        clean_lr = clean_lr.dropna(subset=['linguistic_richness_mean'])
        if len(clean_lr) > 0:
            shared_ceiling = np.log(4)
            print("\n📖 LINGUISTIC RICHNESS (clean data, higher = more relational diversity per rule):")
            print(f"    Shared absolute scale across ALL models — ceiling = log(4) = {shared_ceiling:.4f} nats.")
            print("    H is directly comparable between models; no per-model renormalization.")
            print("-"*80)
            for _, row in clean_lr.sort_values('linguistic_richness_mean', ascending=False).iterrows():
                pct_mean = 100 * row['linguistic_richness_mean'] / shared_ceiling
                pct_std = 100 * row['linguistic_richness_std'] / shared_ceiling
                print(f"  • {row['model'].upper():<15} "
                      f"H = {row['linguistic_richness_mean']:.4f} ± {row['linguistic_richness_std']:.4f} nats  "
                      f"({pct_mean:5.1f}% ± {pct_std:4.1f}% of shared ceiling)")
        
        best_model, details = self.get_best_model()
        print("\n🏆 VERDICT:")
        print("-"*80)
        print(f"✅ BEST MODEL: {best_model}")
        if 'decision_reason' in details:
            print(f"📝 Reason: {details['decision_reason']}")
        
        if 'all_combined_scores' in details:
            print(f"\nAll combined robustness scores:")
            for name, score in sorted(details['all_combined_scores'].items(), key=lambda x: -x[1]):
                print(f"  • {name.upper()}: {score:.4f}")
        elif 'clean_performance' in details:
            print(f"\nClean-data performance summary:")
            for name, perf in details['clean_performance'].items():
                print(f"  • {name.upper()}: Acc={perf['accuracy']:.4f} ± {perf['accuracy_std']:.4f}, "
                      f"AUC={perf['auc']:.4f} ± {perf['auc_std']:.4f}")
        
        print("="*80)