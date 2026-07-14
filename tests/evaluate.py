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
import secrets

from train.early_stop import EarlyStopping
from utils.paper_plot_style import (
    DOUBLE_COLUMN_WIDTH,
    LEGEND_SIZE,
    TITLE_SIZE,
    apply_paper_style,
    make_legend_compact,
    panel_label,
    save_paper_figure,
    style_axis,
)
from utils.plot_style import (
    NEGATIVE_BAR,
    NEUTRAL_LINE,
    POSITIVE_BAR,
    TICK_COLOR,
    apply_plot_style,
    is_highlight_model,
    model_color,
    model_display_name,
    model_line_kwargs,
    plot_bubble_chart,
    save_figure,
    style_axes,
    style_legend,
)

warnings.filterwarnings('ignore')


class Evaluator:
    """
    Flexible robustness evaluator that can compare any number of models with optional noise.

    Each of the ``n_runs`` repeats training from scratch on a *fresh* 70/30
    train/test split (via ``experiment.resplit(session_id=run_seed)``), so reported
    mean ± std reflects both split and initialization variance.
    """
    
    def __init__(self, 
                 experiment,
                 model_configs: Dict[str, Dict], 
                 learning_params: Dict,
                 device: torch.device,
                 binary: bool = True,
                 noise_levels: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                 n_runs: int = 3,
                 random_state: Optional[int] = 42,
                 use_noise: bool = True,
                 use_early_stopping: Optional[bool] = None,
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
        self.use_early_stopping = use_early_stopping
        self.noise_type = noise_type
        
        # Validate noise type
        valid_noise_types = ["gaussian", "salt_pepper"]
        if noise_type not in valid_noise_types:
            raise ValueError(f"noise_type must be one of {valid_noise_types}, got '{noise_type}'")
        
        # Dynamically create results storage for all models
        self.results = {}
        for model_name in model_configs.keys():
            self.results[model_name] = {
                nl: {'train_acc': [], 'test_acc': [], 'train_auc': [], 'test_auc': [],
                     'linguistic_richness': [], 'relaxation_rate': []}
                for nl in self.noise_levels
            }

        # Most recently trained wrapper for each model_name (overwritten
        # every run). Used only to show the rule-by-rule relaxation_rate
        # breakdown in print_detailed_report() — the run-to-run *scalar*
        # average already lives in self.results / summary_df like every
        # other metric.
        self._last_wrapper = {}

        self._use_kfold = getattr(self.experiment, 'is_kfold', lambda: False)()

        X_sample, _ = self.experiment.train_numpy()
        self.n_features = X_sample.shape[1]

        df, target = self.experiment.get_data()
        if isinstance(target, str):
            y_full = df[target]
        elif isinstance(target, int):
            y_full = df.iloc[:, target]
        else:
            y_full = target
        self.n_classes = len(np.unique(np.ravel(y_full)))

        # Populated afresh at the start of every run via _refresh_split().
        self.X_train = self.X_test = self.X_val = None
        self.y_train = self.y_test = self.y_val = None

    @staticmethod
    def _labels_1d(y: np.ndarray) -> np.ndarray:
        if y.ndim == 2:
            return np.argmax(y, axis=1)
        return np.ravel(y)

    def _run_seed(self, run: int, noise_std: float) -> int:
        """Per-run training seed. Fixed when random_state is set; OS-random otherwise."""
        if self.random_state is not None:
            return self.random_state + run * 100 + int(noise_std * 1000)
        return secrets.randbelow(2**32 - 1)

    def _set_run_seed(self, run_seed: int) -> None:
        np.random.seed(run_seed)
        random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

    def _refresh_split(self, session_id: int, fold_index: Optional[int] = None) -> None:
        if self._use_kfold:
            if fold_index is not None:
                self.experiment.set_fold(fold_index)
            else:
                self.experiment.resplit(session_id=session_id)
        else:
            self.experiment.resplit(session_id=session_id)
        self.X_train, self.y_train = self.experiment.train_numpy()
        self.X_test, self.y_test = self.experiment.test_numpy()
        self.y_train = self._labels_1d(self.y_train)
        self.y_test = self._labels_1d(self.y_test)
        if getattr(self.experiment, 'has_validation_split', lambda: False)():
            self.X_val, self.y_val = self.experiment.val_numpy()
            self.y_val = self._labels_1d(self.y_val)
        else:
            self.X_val = self.y_val = None

    def _split_description(self) -> str:
        if hasattr(self.experiment, 'split_description'):
            return self.experiment.split_description()
        return "fresh 70/30 redrawn each run (PyCaret session_id)"

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
                           run_seed: int = 42,
                           fold_index: Optional[int] = None) -> Tuple[float, float, float, float, float, float, Any]:
        """
        Train and evaluate a single model with given noise level
        Returns: (train_acc, test_acc, train_auc, test_auc, linguistic_richness, relaxation_rate, trained_wrapper)

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

        relaxation_rate is the mean per-rule "don't care" relaxation rate,
        in [0, 1], for every model class that exposes
        `model.relaxation_rate()` (UNFIS, GRIFFIN, LitAnfis, GIFTSHIFTER).
        It measures how much a rule leans on a relaxation/"don't care"
        gate rather than committing to its raw membership:
          - A classical model with no such gate (e.g. plain ANFIS) has
            no relaxation_rate at all -> np.nan (not applicable).
          - LitAnfis has only a relational equal/not-equal gate, no
            relaxation gate -> structurally 0.0.
          - UNFIS / GRIFFIN / GIFTSHIFTER each expose a real relaxation
            gate -> can be non-zero, learned from data.
        Like linguistic_richness, np.nan is reserved for "not applicable"
        or "failed run", never used to mean "exactly zero relaxation".
        """
        self._refresh_split(session_id=run_seed, fold_index=fold_index)

        # Feature count can change across resplits when PyCaret fits encoders on
        # train only (e.g. rare categorical levels in Autism's country column).
        run_params = model_params.copy()
        run_params['in_features'] = self.X_train.shape[1]

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
        g.manual_seed(run_seed)

        train_loader = DataLoader(train_dataset, batch_size=self.learning_params['batch_size'], shuffle=True,
                                  worker_init_fn=seed_worker, generator=g)
        
        # Initialize model
        model = model_class(**run_params, dtype=torch.float32)
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
        
        # Training loop — early stopping only when a validation split exists and
        # use_early_stopping is not explicitly disabled (default: auto).
        use_val_for_training = (
            self.X_val is not None
            and (self.use_early_stopping is None or self.use_early_stopping)
        )
        early_stopping = (
            EarlyStopping(patience=10, delta=-0.00001) if use_val_for_training else None
        )

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

            if early_stopping is not None:
                model.eval()
                X_val_noisy = self._apply_noise(self.X_val, noise_std, self.noise_type)
                val_tensor_x = torch.tensor(X_val_noisy, dtype=torch.float32, device=self.device)
                if self.binary:
                    val_tensor_y = torch.tensor(self.y_val, dtype=torch.float32, device=self.device)
                else:
                    val_tensor_y = torch.tensor(self.y_val, dtype=torch.long, device=self.device)

                with torch.no_grad():
                    val_outputs = model(val_tensor_x)
                    val_preds = val_outputs[0]
                    if self.binary:
                        val_loss = cross(val_preds.squeeze(), val_tensor_y.squeeze())
                    else:
                        val_loss = cross(val_preds, val_tensor_y.long())

                early_stopping(val_loss.item(), model)
                if early_stopping.early_stop:
                    break
        
        if early_stopping is not None and early_stopping.best_model_state is not None:
            early_stopping.load_best_model(model)

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
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, wrapper
        
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

        # Relaxation rate (mean across rules, in [0, 1]) — see docstring
        # above. relaxation_rate_per_rule (or None if not applicable) is
        # stashed on the wrapper so evaluate() can keep a rule-by-rule
        # breakdown of the most recent run for print_detailed_report(),
        # alongside the run-averaged scalar in self.results / summary_df.
        if hasattr(model, 'relaxation_rate'):
            try:
                relaxation_rate, relaxation_rate_per_rule = model.relaxation_rate(per_rule=True)
                relaxation_rate_per_rule = relaxation_rate_per_rule.detach().cpu().numpy()
            except Exception as e:
                print(f"Warning: Could not compute relaxation_rate - {e}")
                relaxation_rate, relaxation_rate_per_rule = np.nan, None
        else:
            relaxation_rate, relaxation_rate_per_rule = np.nan, None
        wrapper.relaxation_rate_per_rule = relaxation_rate_per_rule
        
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
                train_auc, test_auc = np.nan, np.nan
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            train_acc, test_acc, train_auc, test_auc = np.nan, np.nan, np.nan, np.nan
            
        return train_acc, test_acc, train_auc, test_auc, linguistic_richness, relaxation_rate, wrapper
    
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
                    model_params['in_features'] = self.n_features
                if 'out_features' not in model_params:
                    if self.binary:
                        model_params['out_features'] = 1
                    else:
                        model_params['out_features'] = self.n_classes
                if 'binary' not in model_params:
                    model_params['binary'] = self.binary
                
                for run in range(self._effective_n_runs()):
                    if verbose:
                        print(f"  {self._run_label(run)}...", end=" ", flush=True)
                    
                    run_seed = self._run_seed(run, noise_std)
                    self._set_run_seed(run_seed)
                    fold_index = run if self._use_kfold else None
                    
                    train_acc, test_acc, train_auc, test_auc, linguistic_richness, relaxation_rate, wrapper = self._train_and_evaluate(
                        config['model_class'],
                        model_params,
                        config['wrapper_class'],
                        noise_std,
                        run,
                        run_seed,
                        fold_index=fold_index,
                    )

                    # Cache the most recently trained wrapper for this model
                    # so print_detailed_report() can show a rule-by-rule
                    # relaxation_rate breakdown (cheap reference, not a copy).
                    self._last_wrapper[model_name] = wrapper

                    if verbose:
                        lr_str = f", LingRich={linguistic_richness:.4f}" if not np.isnan(linguistic_richness) else ""
                        rr_str = f", RelaxRate={relaxation_rate:.4f}" if not np.isnan(relaxation_rate) else ""
                        print(f"Acc={test_acc:.4f}{lr_str}{rr_str}")
                    
                    self.results[model_name][noise_std]['train_acc'].append(train_acc)
                    self.results[model_name][noise_std]['test_acc'].append(test_acc)
                    self.results[model_name][noise_std]['train_auc'].append(train_auc)
                    self.results[model_name][noise_std]['test_auc'].append(test_auc)
                    self.results[model_name][noise_std]['linguistic_richness'].append(linguistic_richness)
                    self.results[model_name][noise_std]['relaxation_rate'].append(relaxation_rate)
                    
                    results_summary.append({
                        'model': model_name,
                        'noise_std': noise_std,
                        'run': run,
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'train_auc': train_auc,
                        'test_auc': test_auc,
                        'linguistic_richness': linguistic_richness,
                        'relaxation_rate': relaxation_rate
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

                    # Same nan-safe aggregation for relaxation_rate (NaN
                    # means "not applicable for this model class", not
                    # "exactly zero relaxation" — see _train_and_evaluate
                    # docstring).
                    rr_values = stats['relaxation_rate']
                    if len(rr_values) > 0 and not all(np.isnan(v) for v in rr_values):
                        rr_mean = np.nanmean(rr_values)
                        rr_std = np.nanstd(rr_values)
                    else:
                        rr_mean, rr_std = np.nan, np.nan
                    
                    summary.append({
                        'model': model_name,
                        'noise_std': noise_std,
                        'train_acc_mean': np.nanmean(stats['train_acc']),
                        'train_acc_std': np.nanstd(stats['train_acc']),
                        'test_acc_mean': np.nanmean(stats['test_acc']),
                        'test_acc_std': np.nanstd(stats['test_acc']),
                        'test_acc_best': np.nanmax(stats['test_acc']),
                        'train_auc_mean': np.nanmean(stats['train_auc']),
                        'train_auc_std': np.nanstd(stats['train_auc']),
                        'test_auc_mean': np.nanmean(stats['test_auc']),
                        'test_auc_std': np.nanstd(stats['test_auc']),
                        'test_auc_best': np.nanmax(stats['test_auc']),
                        'linguistic_richness_mean': lr_mean,
                        'linguistic_richness_std': lr_std,
                        'relaxation_rate_mean': rr_mean,
                        'relaxation_rate_std': rr_std
                    })
        
        return pd.DataFrame(summary)
    
    def plot_robustness_curves(self, save_path: Optional[str] = None):
        """Plot robustness curves for all models"""
        apply_paper_style()
        summary_df = self.get_summary_statistics()
        
        if len(summary_df) == 0:
            print("No data to plot. Run evaluate() first.")
            return
        
        fig, axes = plt.subplots(
            1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.2), constrained_layout=True,
        )
        x_label = "Noise standard deviation (σ)" if self.use_noise else "Experiment"
        model_names = sorted(
            self.model_configs.keys(),
            key=lambda m: (is_highlight_model(m), m),
        )

        for ax, metric, ylabel, panel in zip(
            axes,
            ("test_acc_mean", "test_auc_mean"),
            ("Test accuracy", "Test AUC"),
            ("(a) Accuracy vs. noise", "(b) AUC vs. noise"),
        ):
            style_axis(ax, grid=True)
            for i, model_name in enumerate(model_names):
                model_data = summary_df[summary_df['model'] == model_name]
                if len(model_data) == 0:
                    continue
                x = model_data['noise_std'].values
                y_mean = model_data[metric].values
                y_std = model_data[f"{metric.replace('_mean', '_std')}"].values
                color = model_color(model_name, i)
                label = model_display_name(model_name)
                if is_highlight_model(model_name):
                    label = f"{label} (proposed)"
                line_kw = model_line_kwargs(model_name)
                ax.plot(
                    x, y_mean, "o-", label=label, color=color,
                    markerfacecolor="white", markeredgewidth=1.0, markeredgecolor=color,
                    **line_kw,
                )
                ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                                color=color, alpha=0.15, zorder=line_kw["zorder"] - 1)

            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            panel_label(ax, panel)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
            legend = ax.legend(
                title="Model", loc="best", fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE,
                framealpha=0.90, borderpad=0.3, labelspacing=0.25,
                handlelength=1.4, handletextpad=0.35,
            )
            make_legend_compact(legend)

        if save_path:
            paths = save_paper_figure(fig, save_path)
            print(f"Saved robustness curves: {', '.join(paths)}")
        plt.show()
    
    def plot_relative_performance(self, save_path: Optional[str] = None):
        """Plot relative performance between models"""
        apply_paper_style()
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
        m1 = model_display_name(model_names[0])
        m2 = model_display_name(model_names[1])
        x_label = "Noise standard deviation (σ)" if self.use_noise else "Experiment"
        
        fig, axes = plt.subplots(
            1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.2), constrained_layout=True,
        )
        
        for ax, values, ylabel, panel in zip(
            axes,
            (acc_improvement, auc_improvement),
            (f"Δ accuracy ({m1} − {m2})", f"Δ AUC ({m1} − {m2})"),
            ("(a) Relative accuracy", "(b) Relative AUC"),
        ):
            style_axis(ax, grid=True)
            colors = [POSITIVE_BAR if v > 0 else NEGATIVE_BAR for v in values]
            bars = ax.bar(
                range(len(x)), values, color=colors, alpha=0.85,
                edgecolor="white", linewidth=0.8, width=0.72,
            )
            ax.axhline(y=0, color=NEUTRAL_LINE, linestyle="-", linewidth=0.9, alpha=0.7)
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels([f"{v:.2f}" for v in x])
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            panel_label(ax, panel)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.3f}"))

            ymax = max(abs(values)) if len(values) else 0.01
            offset = 0.04 * ymax if ymax > 0 else 0.01
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + offset if val >= 0 else height - offset,
                    f"{val:+.3f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8.5, color=TICK_COLOR,
                )

        if save_path:
            paths = save_paper_figure(fig, save_path)
            print(f"Saved relative performance: {', '.join(paths)}")
        plt.show()

    def plot_model_landscape(
        self,
        save_path: Optional[str] = None,
        interactive: bool = False,
        noise_std: Optional[float] = None,
    ):
        """Bubble chart: accuracy vs. AUC (clean data), bubble size = linguistic richness."""
        summary_df = self.get_summary_statistics()
        if len(summary_df) == 0:
            print("No data to plot. Run evaluate() first.")
            return

        if noise_std is None:
            noise_std = self._clean_noise_level()
        clean = summary_df[np.isclose(summary_df["noise_std"], noise_std)].copy()
        if len(clean) == 0:
            print("No clean-data summary available for model landscape.")
            return
        clean["linguistic_richness_mean"] = clean["linguistic_richness_mean"].fillna(0)
        clean["relaxation_rate_mean"] = clean["relaxation_rate_mean"].fillna(0)

        plot_bubble_chart(
            clean,
            x="test_acc_mean",
            y="test_auc_mean",
            size="linguistic_richness_mean",
            color="relaxation_rate_mean",
            xerr="test_acc_std",
            yerr="test_auc_std",
            xlabel="Test accuracy",
            ylabel="Test AUC",
            size_label="Linguistic richness",
            color_label="Relaxation rate",
            save_path=save_path,
            interactive=interactive,
        )

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

    def _effective_n_runs(self) -> int:
        if self._use_kfold:
            return self.experiment.n_folds
        return self.n_runs

    def _run_label(self, run: int) -> str:
        if self._use_kfold:
            return f"Fold {run + 1}/{self.experiment.n_folds}"
        return f"Run {run + 1}/{self.n_runs}"

    def _aggregation_note(self) -> str:
        if self._use_kfold:
            return f"Number of folds: {self.experiment.n_folds} (mean ± std across folds)"
        return f"Number of runs per configuration: {self.n_runs}"

    def _summary_metric_suffix(self) -> str:
        """Suffix for summary column headers (e.g. ' (top-30)')."""
        return ""

    def _summary_column_labels(self) -> Dict[str, str]:
        suffix = self._summary_metric_suffix()
        return {
            "test_acc": "Test Acc ± Std" + suffix,
            "test_auc": "Test AUC ± Std" + suffix,
            "linguistic_richness": "Ling. Richness ± Std" + suffix,
            "relaxation_rate": "Relax. Rate ± Std" + suffix,
        }

    def _show_best_in_summary_table(self) -> bool:
        """If False, Best Acc/AUC are omitted from the main summary table."""
        return True

    def _clean_noise_level(self) -> float:
        return 0.0 if 0.0 in self.noise_levels else self.noise_levels[0]

    def _print_best_run_metrics(self, summary_df: pd.DataFrame) -> None:
        """Print peak single-run test metrics (may differ from summary averages)."""
        clean_best = summary_df[summary_df['noise_std'] == self._clean_noise_level()]
        if len(clean_best) == 0:
            return
        print("\n🏅 BEST SINGLE-RUN TEST METRICS (max across runs, clean data):")
        print("-"*80)
        for _, row in clean_best.sort_values('test_acc_best', ascending=False).iterrows():
            acc_best = "N/A" if np.isnan(row['test_acc_best']) else f"{row['test_acc_best']:.4f}"
            auc_best = "N/A" if np.isnan(row['test_auc_best']) else f"{row['test_auc_best']:.4f}"
            print(f"  • {row['model'].upper():<15} Best Test Acc = {acc_best}, Best Test AUC = {auc_best}")
    
    def print_detailed_report(self):
        """Print a detailed report of the evaluation"""
        print("\n" + "="*80)
        print("MODEL EVALUATION REPORT")
        print(f"Dataset: {self.experiment.__class__.__name__}")
        print(f"Noise Enabled: {self.use_noise}")
        if self.use_noise:
            print(f"Noise Type: {self.noise_type}")
        print(self._aggregation_note())
        print(f"Train/validation/test split: {self._split_description()}")
        print("="*80)
        
        summary_df = self.get_summary_statistics()
        if len(summary_df) > 0:
            print("\n📊 PERFORMANCE SUMMARY:")
            print("-"*145)
            show_best = self._show_best_in_summary_table()
            cols = self._summary_column_labels()
            if show_best:
                print(f"{'Model':<15} {'Noise σ':<10} {cols['test_acc']:<32} {'Best Acc':<10} "
                      f"{cols['test_auc']:<32} {'Best AUC':<10} {cols['linguistic_richness']:<34} {cols['relaxation_rate']:<28}")
            else:
                print(f"{'Model':<15} {'Noise σ':<10} {cols['test_acc']:<32} "
                      f"{cols['test_auc']:<32} {cols['linguistic_richness']:<34} {cols['relaxation_rate']:<28}")
            print("-"*145)
            
            for model_name in self.model_configs.keys():
                model_data = summary_df[summary_df['model'] == model_name]
                for _, row in model_data.iterrows():
                    noise_label = f"{row['noise_std']:.2f}" if self.use_noise else "Clean"
                    best_acc_label = "N/A" if np.isnan(row['test_acc_best']) else f"{row['test_acc_best']:.4f}"
                    best_auc_label = "N/A" if np.isnan(row['test_auc_best']) else f"{row['test_auc_best']:.4f}"
                    if np.isnan(row['linguistic_richness_mean']):
                        lr_label = "N/A"
                    else:
                        lr_label = f"{row['linguistic_richness_mean']:.4f} ± {row['linguistic_richness_std']:.4f}"
                    if np.isnan(row['relaxation_rate_mean']):
                        rr_label = "N/A"
                    else:
                        rr_label = f"{row['relaxation_rate_mean']:.4f} ± {row['relaxation_rate_std']:.4f}"
                    if show_best:
                        print(f"{model_name.upper():<15} {noise_label:<10} "
                              f"{row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f}    "
                              f"{best_acc_label:<10} "
                              f"{row['test_auc_mean']:.4f} ± {row['test_auc_std']:.4f}    "
                              f"{best_auc_label:<10} "
                              f"{lr_label:<22} {rr_label:<20}")
                    else:
                        print(f"{model_name.upper():<15} {noise_label:<10} "
                              f"{row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f}    "
                              f"{row['test_auc_mean']:.4f} ± {row['test_auc_std']:.4f}    "
                              f"{lr_label:<26} {rr_label:<24}")
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

        # Relaxation rate: descriptive only (not folded into best-model
        # selection), shown on the clean-data (noise_std == 0) split for
        # the subset of models that expose model.relaxation_rate().
        #
        # Unlike linguistic_richness (which measures *which* relation a
        # rule expresses), relaxation_rate measures how much a rule
        # leans on a "don't care" gate instead of committing to its raw
        # membership at all. It is in [0, 1], comparable across model
        # classes with no renormalization:
        #   - A classical rule with no such gate is 0.0 by construction
        #     (ANFIS has no relaxation_rate at all -> NaN/not applicable;
        #     LitAnfis has only a relational gate, no relaxation gate at
        #     all -> structurally 0.0).
        #   - UNFIS / GRIFFIN / GIFTSHIFTER each expose a real, learned
        #     relaxation gate and can land anywhere in (0, 1].
        #
        # Reported two ways, as requested: the rules-average scalar
        # (from the run-averaged summary_df, same aggregation as every
        # other metric) AND a rule-by-rule breakdown taken from the most
        # recently trained model for each model_name.
        clean_rr = summary_df[summary_df['noise_std'] == 0] if self.use_noise else summary_df
        clean_rr = clean_rr.dropna(subset=['relaxation_rate_mean'])
        if len(clean_rr) > 0:
            print("\n🌊 RELAXATION RATE (clean data, higher = rules rely more on a \"don't care\" gate):")
            print("    Scale [0, 1], comparable across models; 0 = no relaxation, no per-model renormalization.")
            print("-"*80)
            for _, row in clean_rr.sort_values('relaxation_rate_mean', ascending=False).iterrows():
                print(f"  • {row['model'].upper():<15} "
                      f"rate = {row['relaxation_rate_mean']:.4f} ± {row['relaxation_rate_std']:.4f}  "
                      f"(rules-average)")

                wrapper = self._last_wrapper.get(row['model'])
                per_rule = getattr(wrapper, 'relaxation_rate_per_rule', None) if wrapper is not None else None
                if per_rule is not None:
                    per_rule_str = ", ".join(f"r{idx}={val:.4f}" for idx, val in enumerate(per_rule))
                    print(f"      per-rule (most recent run): {per_rule_str}")

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

        self._print_best_run_metrics(summary_df)
        
        print("="*80)


class TopKEvaluator(Evaluator):
    """
    Run the model ``n_runs`` times with random seeds, then report mean ± std
    over the ``top_k`` best runs (ranked by ``ranking_metric``, default test_acc).

    All other behaviour (noise sweeps, plots, best-model selection) uses the
    top-k aggregated statistics from :meth:`get_summary_statistics`.
    """

    VALID_RANKING_METRICS = ("test_acc", "test_auc", "both")

    def __init__(
        self,
        *args,
        top_k: int = 30,
        ranking_metric: str = "both",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if ranking_metric not in self.VALID_RANKING_METRICS:
            raise ValueError(
                f"ranking_metric must be one of {self.VALID_RANKING_METRICS}, "
                f"got '{ranking_metric}'"
            )
        self.top_k = top_k
        self.ranking_metric = ranking_metric

    def _aggregation_note(self) -> str:
        if self._use_kfold:
            return (
                f"Total folds: {self.experiment.n_folds} "
                f"(mean ± std over all folds)"
            )
        if self.ranking_metric == "both":
            return (
                f"Total runs per configuration: {self.n_runs} "
                f"(mean ± std over top {self.top_k} runs: "
                f"acc ranked by test_acc, auc ranked by test_auc)"
            )
        return (
            f"Total runs per configuration: {self.n_runs} "
            f"(mean ± std over top {self.top_k} runs by {self.ranking_metric})"
        )

    def _summary_metric_suffix(self) -> str:
        return f" (top-{self.top_k} runs)"

    def _summary_column_labels(self) -> Dict[str, str]:
        """Per-column header labels for the performance summary table."""
        k = self.top_k
        if self.ranking_metric == "both":
            return {
                "test_acc": f"Test Acc ± Std (top-{k} runs, by acc)",
                "test_auc": f"Test AUC ± Std (top-{k} runs, by auc)",
                "linguistic_richness": f"Ling. Richness ± Std (top-{k} runs, by acc)",
                "relaxation_rate": f"Relax. Rate ± Std (top-{k} runs, by acc)",
            }
        suffix = self._summary_metric_suffix()
        return {
            "test_acc": "Test Acc ± Std" + suffix,
            "test_auc": "Test AUC ± Std" + suffix,
            "linguistic_richness": "Ling. Richness ± Std" + suffix,
            "relaxation_rate": "Relax. Rate ± Std" + suffix,
        }

    def _show_best_in_summary_table(self) -> bool:
        return False

    def get_peak_run_metrics(self, noise_std: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """
        Peak single-run test acc and test auc per model on clean data.
        Acc and auc peaks may come from different runs.
        """
        if noise_std is None:
            noise_std = self._clean_noise_level()
        peaks: Dict[str, Dict[str, Any]] = {}
        for model_name in self.model_configs:
            stats = self.results[model_name][noise_std]
            acc_arr = np.asarray(stats["test_acc"], dtype=float)
            auc_arr = np.asarray(stats["test_auc"], dtype=float)
            if len(acc_arr) == 0:
                continue
            entry: Dict[str, Any] = {"model": model_name, "noise_std": noise_std}
            if not all(np.isnan(acc_arr)):
                acc_idx = int(np.nanargmax(acc_arr))
                entry["test_acc_best"] = float(acc_arr[acc_idx])
                entry["test_acc_best_run"] = acc_idx + 1
            else:
                entry["test_acc_best"] = np.nan
                entry["test_acc_best_run"] = None
            if not all(np.isnan(auc_arr)):
                auc_idx = int(np.nanargmax(auc_arr))
                entry["test_auc_best"] = float(auc_arr[auc_idx])
                entry["test_auc_best_run"] = auc_idx + 1
            else:
                entry["test_auc_best"] = np.nan
                entry["test_auc_best_run"] = None
            peaks[model_name] = entry
        return peaks

    def _print_best_run_metrics(self, summary_df: pd.DataFrame) -> None:
        peaks = self.get_peak_run_metrics()
        if not peaks:
            return
        print("\n🏅 PEAK SINGLE-RUN SCORES (not top-k averaged — one best acc, one best auc per model):")
        print(f"    Across all {self.n_runs} runs on clean data; acc and auc peaks may be from different runs.")
        print("-"*80)
        for model_name in self.model_configs.keys():
            if model_name not in peaks:
                continue
            p = peaks[model_name]
            print(f"  • {model_name.upper()}")
            acc = p["test_acc_best"]
            auc = p["test_auc_best"]
            if not np.isnan(acc):
                print(f"      Best Test Acc : {acc:.4f}   (run {p['test_acc_best_run']}/{self.n_runs})")
            else:
                print("      Best Test Acc : N/A")
            if not np.isnan(auc):
                print(f"      Best Test AUC : {auc:.4f}   (run {p['test_auc_best_run']}/{self.n_runs})")
            else:
                print("      Best Test AUC : N/A")

    @staticmethod
    def _top_k_indices(ranking_values: List[float], top_k: int) -> List[int]:
        """Indices of the top_k runs; NaN scores are excluded from selection."""
        arr = np.asarray(ranking_values, dtype=float)
        if len(arr) == 0:
            return []
        k = min(top_k, len(arr))
        valid = np.where(~np.isnan(arr))[0]
        if len(valid) == 0:
            return list(range(k))
        order = valid[np.argsort(-arr[valid])]
        return order[:k].tolist()

    def _subset_top_k(self, stats: Dict[str, List], ranking_metric: Optional[str] = None) -> Dict[str, List]:
        """Keep only the top_k runs (by ranking_metric) for each metric list."""
        metric = ranking_metric or self.ranking_metric
        top_indices = self._top_k_indices(stats[metric], self.top_k)
        return {key: [vals[i] for i in top_indices] for key, vals in stats.items()}

    def _subset_top_k_for_summary(self, all_stats: Dict[str, List]) -> Dict[str, List]:
        """
        Build per-metric top-k subsets. With ranking_metric='both', acc-related
        metrics use top-k by test_acc and auc-related metrics use top-k by test_auc.
        """
        if self.ranking_metric != "both":
            return self._subset_top_k(all_stats)

        acc_runs = self._subset_top_k(all_stats, ranking_metric="test_acc")
        auc_runs = self._subset_top_k(all_stats, ranking_metric="test_auc")
        return {
            "train_acc": acc_runs["train_acc"],
            "test_acc": acc_runs["test_acc"],
            "train_auc": auc_runs["train_auc"],
            "test_auc": auc_runs["test_auc"],
            "linguistic_richness": acc_runs["linguistic_richness"],
            "relaxation_rate": acc_runs["relaxation_rate"],
        }

    def get_summary_statistics(self) -> pd.DataFrame:
        """Summary statistics over the top_k best runs per model/noise level."""
        summary = []

        for model_name in self.model_configs.keys():
            for noise_std in self.noise_levels:
                all_stats = self.results[model_name][noise_std]
                stats = self._subset_top_k_for_summary(all_stats)
                if len(stats["test_acc"]) > 0:
                    lr_values = stats["linguistic_richness"]
                    if len(lr_values) > 0 and not all(np.isnan(v) for v in lr_values):
                        lr_mean = np.nanmean(lr_values)
                        lr_std = np.nanstd(lr_values)
                    else:
                        lr_mean, lr_std = np.nan, np.nan

                    rr_values = stats["relaxation_rate"]
                    if len(rr_values) > 0 and not all(np.isnan(v) for v in rr_values):
                        rr_mean = np.nanmean(rr_values)
                        rr_std = np.nanstd(rr_values)
                    else:
                        rr_mean, rr_std = np.nan, np.nan

                    summary.append({
                        "model": model_name,
                        "noise_std": noise_std,
                        "train_acc_mean": np.nanmean(stats["train_acc"]),
                        "train_acc_std": np.nanstd(stats["train_acc"]),
                        "test_acc_mean": np.nanmean(stats["test_acc"]),
                        "test_acc_std": np.nanstd(stats["test_acc"]),
                        "test_acc_best": np.nanmax(all_stats["test_acc"]),
                        "train_auc_mean": np.nanmean(stats["train_auc"]),
                        "train_auc_std": np.nanstd(stats["train_auc"]),
                        "test_auc_mean": np.nanmean(stats["test_auc"]),
                        "test_auc_std": np.nanstd(stats["test_auc"]),
                        "test_auc_best": np.nanmax(all_stats["test_auc"]),
                        "linguistic_richness_mean": lr_mean,
                        "linguistic_richness_std": lr_std,
                        "relaxation_rate_mean": rr_mean,
                        "relaxation_rate_std": rr_std,
                    })

        return pd.DataFrame(summary)