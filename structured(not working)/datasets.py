from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, Literal

import numpy as np
from sklearn.datasets import make_circles, make_moons, make_swiss_roll
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DatasetSplits = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
TaskType = Literal["binary_classification", "multiclass_classification", "regression"]

if TYPE_CHECKING:
    import torch


class BaseSyntheticDataset:
    """Common utilities for synthetic classification datasets."""

    target_type: str = "classification"

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        sample_seed: Optional[int] = None,
        stratify: bool = True,
        shuffle: bool = True,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.sample_seed = sample_seed if sample_seed is not None else random_state
        self.stratify = stratify
        self.shuffle = shuffle
        self._splits: Optional[DatasetSplits] = None
        self.num_classes: Optional[int] = None
        self.task_type: Optional[TaskType] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def numpy(self) -> DatasetSplits:
        """Return (X_train, X_test, y_train, y_test) as numpy arrays."""
        return self._ensure_splits()

    def torch_tensors(
        self,
        device: Optional["torch.device | str"] = None,
        x_dtype: Optional["torch.dtype"] = None,
        y_dtype: Optional["torch.dtype"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return tensors suitable for PyTorch training loops."""
        import torch

        x_dtype = x_dtype or torch.float32
        X_train, X_test, y_train, y_test = self._ensure_splits()

        if y_dtype is None:
            y_dtype = torch.float32 if y_train.dtype.kind == "f" else torch.long

        return (
            torch.tensor(X_train, dtype=x_dtype, device=device),
            torch.tensor(X_test, dtype=x_dtype, device=device),
            torch.tensor(y_train, dtype=y_dtype, device=device),
            torch.tensor(y_test, dtype=y_dtype, device=device),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_splits(self) -> DatasetSplits:
        if self._splits is None:
            X, y = self._generate()
            stratify = y if self.stratify and self.target_type == "classification" else None
            splits = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify,
            )
            self._splits = (
                splits[0].astype(np.float32),
                splits[1].astype(np.float32),
                splits[2].astype(y.dtype),
                splits[3].astype(y.dtype),
            )
        return self._splits

    def _generate(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# -------------------------------------------------------------------------
# Specific datasets
# -------------------------------------------------------------------------


class MoonsDataset(BaseSyntheticDataset):
    """Two interleaving moons from sklearn."""

    def __init__(
        self,
        n_samples: int = 2000,
        noise: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.noise = noise
        self.num_classes = 2
        self.task_type = "binary_classification"

    def _generate(self) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_moons(
            n_samples=self.n_samples,
            noise=self.noise,
            random_state=self.sample_seed,
        )
        return X.astype(np.float32), y.astype(np.int64)


class CirclesDataset(BaseSyntheticDataset):
    """Concentric circles with optional scaling."""

    def __init__(
        self,
        n_samples: int = 2000,
        noise: float = 0.05,
        factor: float = 0.4,
        scale: bool = True,
        test_size: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(test_size=test_size, **kwargs)
        self.n_samples = n_samples
        self.noise = noise
        self.factor = factor
        self.scale = scale
        self.num_classes = 2
        self.task_type = "binary_classification"

    def _generate(self) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_circles(
            n_samples=self.n_samples,
            noise=self.noise,
            factor=self.factor,
            random_state=self.sample_seed,
        )
        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        return X.astype(np.float32), y.astype(np.int64)


class SpiralDataset(BaseSyntheticDataset):
    """Two-class spiral dataset (hand-crafted)."""

    def __init__(
        self,
        points_per_class: int = 2000,
        noise: float = 0.2,
        rotation: float = 780 * (2 * np.pi) / 360,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.points_per_class = points_per_class
        self.noise = noise
        self.rotation = rotation
        self.num_classes = 2
        self.task_type = "binary_classification"

    def _generate(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.sample_seed)
        n = np.sqrt(rng.random((self.points_per_class, 1))) * self.rotation
        d1x = -np.cos(n) * n + rng.random((self.points_per_class, 1)) * self.noise
        d1y = np.sin(n) * n + rng.random((self.points_per_class, 1)) * self.noise
        X = np.vstack(
            (
                np.hstack((d1x, d1y)),
                np.hstack((-d1x, -d1y)),
            )
        )
        y = np.hstack(
            (
                np.zeros(self.points_per_class, dtype=np.int64),
                np.ones(self.points_per_class, dtype=np.int64),
            )
        )
        return X.astype(np.float32), y


class SwissRollDataset(BaseSyntheticDataset):
    """Swiss-roll in 3D with configurable label granularity."""

    def __init__(
        self,
        n_samples: int = 3000,
        noise: float = 0.4,
        label_mode: str = "ternary",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.noise = noise
        self.label_mode = label_mode
        self.task_type = self._infer_task_type()
        self.num_classes = self._infer_num_classes()

    def _infer_num_classes(self) -> Optional[int]:
        if self.label_mode == "binary":
            return 2
        if self.label_mode == "ternary":
            return 3
        return None

    def _infer_task_type(self) -> TaskType:
        if self.label_mode == "binary":
            return "binary_classification"
        if self.label_mode == "ternary":
            return "multiclass_classification"
        return "regression"

    def _generate(self) -> Tuple[np.ndarray, np.ndarray]:
        X3d, t = make_swiss_roll(
            n_samples=self.n_samples,
            noise=self.noise,
            random_state=self.sample_seed,
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X3d).astype(np.float32)

        if self.label_mode == "binary":
            labels = (t > np.median(t)).astype(np.int64)
        elif self.label_mode == "ternary":
            tertiles = np.quantile(t, [1 / 3, 2 / 3])
            labels = np.digitize(t, tertiles).astype(np.int64)
        else:
            labels = t.astype(np.float32)
            self.target_type = "regression"
            self.stratify = False
            self.task_type = "regression"

        if self.target_type == "classification":
            labels = labels.astype(np.int64)
        return X_scaled, labels


__all__ = [
    "BaseSyntheticDataset",
    "MoonsDataset",
    "CirclesDataset",
    "SpiralDataset",
    "SwissRollDataset",
]