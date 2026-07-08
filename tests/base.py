import pandas as pd
import numpy as np
import torch
from typing import List, Optional, Tuple
from torch.utils.data import TensorDataset
from pycaret.classification import ClassificationExperiment
from sklearn.model_selection import StratifiedKFold, train_test_split


class Test:
    TRAIN_SIZE = 0.7  # 70% train, 30% test (default two-way split)
    SPLIT_MODES = ("holdout", "kfold")

    def __init__(self, *args, **kwargs) -> None:
        test_data = kwargs.pop('test_data', None)
        self.split_mode = kwargs.pop('split_mode', 'holdout')
        if self.split_mode not in self.SPLIT_MODES:
            raise ValueError(
                f"split_mode must be one of {self.SPLIT_MODES}, got '{self.split_mode}'"
            )
        self.n_folds = int(kwargs.pop('n_folds', 10))
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")

        self.split_ratios = kwargs.pop('split_ratios', None)
        if self.split_ratios is not None:
            ratios = tuple(float(r) for r in self.split_ratios)
            if len(ratios) not in (2, 3):
                raise ValueError(
                    "split_ratios must be a 2-tuple (train, test) or "
                    "3-tuple (train, val, test)"
                )
            if not np.isclose(sum(ratios), 1.0):
                raise ValueError(f"split_ratios must sum to 1.0, got {sum(ratios)}")
            self.split_ratios = ratios
            if self.split_mode == 'kfold':
                expected_test = 1.0 / self.n_folds
                test_frac = ratios[-1]
                if not np.isclose(test_frac, expected_test):
                    raise ValueError(
                        f"k-fold split_ratios test fraction {test_frac:.4f} does not "
                        f"match 1/n_folds={expected_test:.4f} for n_folds={self.n_folds}"
                    )

        if test_data is not None:
            self.df = pd.concat([self.df, test_data], ignore_index=True)

        self._setup_args = args
        self._setup_kwargs = kwargs
        self._val_features = None
        self._val_target = None
        self._fold_splits: Optional[List[Tuple[np.ndarray, ...]]] = None
        self._current_fold = 0
        self._cv_seed: Optional[int] = None
        self._setup()

    def _target_series(self) -> pd.Series:
        if isinstance(self.target, str):
            return self.df[self.target]
        if isinstance(self.target, int):
            return self.df.iloc[:, self.target]
        return pd.Series(self.target, index=self.df.index)

    def _target_name(self) -> str:
        if isinstance(self.target, str):
            return self.target
        if isinstance(self.target, int):
            return self.df.columns[self.target]
        return self._target_series().name

    @staticmethod
    def _stratified_three_way_indices(
        y: np.ndarray,
        session_id: int,
        train_frac: float,
        val_frac: float,
        test_frac: float,
    ):
        idx = np.arange(len(y))
        holdout_frac = val_frac + test_frac
        train_idx, holdout_idx = train_test_split(
            idx,
            test_size=holdout_frac,
            stratify=y,
            random_state=session_id,
        )
        test_share_of_holdout = test_frac / holdout_frac
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=test_share_of_holdout,
            stratify=y[holdout_idx],
            random_state=session_id,
        )
        return train_idx, val_idx, test_idx

    def _compute_fold_splits(self, y: np.ndarray, seed: int) -> List[Tuple[np.ndarray, ...]]:
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=seed,
        )
        splits: List[Tuple[np.ndarray, ...]] = []
        for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
            if self.split_ratios is not None and len(self.split_ratios) == 3:
                train_frac, val_frac, _test_frac = self.split_ratios
                val_share_of_train = val_frac / train_frac
                sub_train_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=val_share_of_train,
                    stratify=y[train_idx],
                    random_state=seed,
                )
                splits.append((sub_train_idx, val_idx, test_idx))
            else:
                splits.append((train_idx, test_idx))
        return splits

    def _setup_from_indices(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        val_idx: Optional[np.ndarray] = None,
    ) -> None:
        """Fit PyCaret preprocessing (imputation, scaling) on train_idx only."""
        kwargs = dict(self._setup_kwargs)
        kwargs.setdefault('verbose', False)
        kwargs.setdefault('imputation_type', 'iterative')
        kwargs.setdefault('normalize', True)
        kwargs.setdefault('index', False)

        setup_data = self.df.iloc[train_idx].reset_index(drop=True)
        setup_target = self._target_series().iloc[train_idx].reset_index(drop=True)
        test_data = self.df.iloc[test_idx].reset_index(drop=True)
        test_target = self._target_series().iloc[test_idx].reset_index(drop=True)

        if val_idx is not None and len(val_idx) > 0:
            self._val_features = self.df.iloc[val_idx].reset_index(drop=True)
            self._val_target = self._target_series().iloc[val_idx].reset_index(drop=True)
        else:
            self._val_features = None
            self._val_target = None

        self.clf = ClassificationExperiment()
        setup_kwargs = dict(
            data=setup_data,
            target=setup_target,
            *self._setup_args,
            **kwargs,
        )
        if isinstance(self.target, str):
            setup_kwargs['test_data'] = test_data
        else:
            test_data_with_target = test_data.copy()
            test_data_with_target[self._target_name()] = test_target.to_numpy()
            setup_kwargs['test_data'] = test_data_with_target
        self.clf.setup(**setup_kwargs)

    def _setup(
        self,
        session_id: Optional[int] = None,
        fold_index: Optional[int] = None,
    ) -> None:
        kwargs = dict(self._setup_kwargs)
        if session_id is not None:
            kwargs['session_id'] = session_id
        kwargs.setdefault('verbose', False)
        kwargs.setdefault('imputation_type', 'iterative')
        kwargs.setdefault('normalize', True)
        kwargs.setdefault('index', False)

        if self.split_mode == 'kfold':
            seed = session_id if session_id is not None else kwargs.get('session_id', 42)
            y = self._target_series().to_numpy()
            if self._fold_splits is None or self._cv_seed != seed:
                self._fold_splits = self._compute_fold_splits(y, seed)
                self._cv_seed = seed
            fold = self._current_fold if fold_index is None else fold_index
            fold = int(fold) % self.n_folds
            self._current_fold = fold
            split = self._fold_splits[fold]
            if len(split) == 3:
                train_idx, val_idx, test_idx = split
            else:
                train_idx, test_idx = split
                val_idx = None
            self._setup_from_indices(train_idx, test_idx, val_idx)
            return

        val_idx = None
        if self.split_ratios is not None:
            seed = session_id if session_id is not None else kwargs.get('session_id', 42)
            y = self._target_series().to_numpy()
            idx = np.arange(len(y))

            if len(self.split_ratios) == 2:
                _train_frac, test_frac = self.split_ratios
                train_idx, test_idx = train_test_split(
                    idx,
                    test_size=test_frac,
                    stratify=y,
                    random_state=seed,
                )
            else:
                train_frac, val_frac, test_frac = self.split_ratios
                train_idx, val_idx, test_idx = self._stratified_three_way_indices(
                    y, seed, train_frac, val_frac, test_frac,
                )
        else:
            kwargs.setdefault('train_size', self.TRAIN_SIZE)
            self.clf = ClassificationExperiment()
            setup_kwargs = dict(
                data=self.df,
                target=self.target,
                *self._setup_args,
                **kwargs,
            )
            self.clf.setup(**setup_kwargs)
            self._val_features = None
            self._val_target = None
            return

        self._setup_from_indices(train_idx, test_idx, val_idx)

    def is_kfold(self) -> bool:
        return self.split_mode == 'kfold'

    def set_fold(self, fold_index: int) -> None:
        """Select a fold for k-fold cross-validation (0 .. n_folds-1)."""
        if not self.is_kfold():
            raise AttributeError("set_fold() is only available when split_mode='kfold'")
        self._setup(fold_index=fold_index)

    def resplit(self, session_id: int) -> None:
        """Re-draw partitions: new random holdout split, or new CV fold shuffle."""
        if self.is_kfold():
            self._fold_splits = None
            self._setup(session_id=session_id, fold_index=self._current_fold)
        else:
            self._setup(session_id=session_id)

    def split_description(self) -> str:
        if self.is_kfold():
            train_frac = (self.n_folds - 1) / self.n_folds
            test_frac = 1.0 / self.n_folds
            if self.split_ratios is not None:
                if len(self.split_ratios) == 3:
                    parts = [int(round(r * 100)) for r in self.split_ratios]
                    label = "/".join(str(p) for p in parts)
                else:
                    parts = [int(round(r * 100)) for r in (train_frac, test_frac)]
                    label = "/".join(str(p) for p in parts)
            else:
                parts = [int(round(train_frac * 100)), int(round(test_frac * 100))]
                label = "/".join(str(p) for p in parts)
            val_note = " with validation carved from train" if self.has_validation_split() else ""
            return (
                f"{self.n_folds}-fold stratified CV, {label} train/test per fold"
                f"{val_note} (StratifiedKFold, session_id)"
            )
        if self.split_ratios is not None:
            parts = [int(round(r * 100)) for r in self.split_ratios]
            label = "/".join(str(p) for p in parts)
            return (
                f"fresh {label} stratified split redrawn each run "
                f"(sklearn train_test_split, session_id)"
            )
        train_pct = int(round(self.TRAIN_SIZE * 100))
        test_pct = 100 - train_pct
        return f"fresh {train_pct}/{test_pct} redrawn each run (PyCaret session_id)"

    def has_validation_split(self) -> bool:
        return self._val_features is not None

    def get_data(self):
        return self.df, self.target

    def _transformed_val_features(self) -> np.ndarray:
        if self._val_features is None:
            raise AttributeError("No validation split configured for this experiment")
        transformed = self.clf.pipeline.transform(self._val_features)
        if hasattr(transformed, 'values'):
            return transformed.values
        return np.asarray(transformed)

    def train_numpy(self):
        return self.clf.X_train_transformed.values, self.clf.y_train_transformed.values

    def val_numpy(self):
        if self._val_features is None:
            raise AttributeError("No validation split configured for this experiment")
        return self._transformed_val_features(), self._val_target.to_numpy()

    def test_numpy(self):
        return self.clf.X_test_transformed.values, self.clf.y_test_transformed.values

    def train_tensor(self, device=None, dtype=torch.float32):
        return Test._tensor(self.train_numpy(), device, dtype)

    def val_tensor(self, device=None, dtype=torch.float32):
        return Test._tensor(self.val_numpy(), device, dtype)

    def test_tensor(self, device=None, dtype=torch.float32):
        return Test._tensor(self.test_numpy(), device, dtype)

    def train_dataset(self, device=None, dtype=torch.float32):
        return Test._dataset(self.train_numpy(), device, dtype)

    def val_dataset(self, device=None, dtype=torch.float32):
        return Test._dataset(self.val_numpy(), device, dtype)

    def test_dataset(self, device=None, dtype=torch.float32):
        return Test._dataset(self.test_numpy(), device, dtype)

    def evaluate_model(self, model):
        self.clf.evaluate_model(model)

    @staticmethod
    def _tensor(data, device=None, dtype=torch.float32):
        device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        return list(map(lambda x: torch.from_numpy(x).to(device).to(dtype), data))

    @staticmethod
    def _dataset(data, device=None, dtype=torch.float32):
        data = Test._tensor(data, device, dtype)
        return TensorDataset(*data)
