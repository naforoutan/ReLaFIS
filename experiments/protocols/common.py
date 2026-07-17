"""Shared helpers for paper protocol runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass
class ProtocolRunResult:
    """Single train/test evaluation under a paper (or generic) protocol."""

    train_acc: float
    test_acc: float
    train_auc: float
    test_auc: float
    linguistic_richness: float
    relaxation_rate: float
    wrapper: Any
    protocol_name: str
    extras: Dict[str, Any] = field(default_factory=dict)


def minmax_fit_transform(
    X_train: np.ndarray,
    *others: Optional[np.ndarray],
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, ...]:
    """Train-only linear scaling into ``feature_range`` (paper-style)."""
    X_train = np.asarray(X_train, dtype=np.float64)
    lo = X_train.min(axis=0)
    hi = X_train.max(axis=0)
    span = hi - lo
    span = np.where(span < 1e-12, 1.0, span)
    low, high = feature_range

    def _scale(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        u = (X - lo) / span
        return (u * (high - low) + low).astype(np.float32)

    out = [_scale(X_train)]
    for X in others:
        out.append(None if X is None else _scale(X))
    return tuple(out)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y).reshape(-1).astype(np.int64, copy=False)
    oh = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    valid = (y >= 0) & (y < n_classes)
    oh[np.where(valid)[0], y[valid]] = 1.0
    return oh


# Aliases used by protocol runners / Evaluator bridges.
one_hot_targets = one_hot


def paper_mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ADMTSK eq. (9): ``(1/(2N)) Σ_n Σ_c (y − z)^2``."""
    n = preds.shape[0]
    return (preds - targets).pow(2).sum() / (2.0 * max(n, 1))


paper_mse_classification_loss = paper_mse_loss


def apply_optional_noise(
    X: np.ndarray,
    noise_std: float,
    apply_noise,
) -> np.ndarray:
    """Apply Evaluator noise after paper scaling when ``noise_std > 0``."""
    if apply_noise is None or not noise_std:
        return X
    return apply_noise(X, float(noise_std))


def labels_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    return np.ravel(y)


def compute_acc_auc(
    wrapper: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    binary: bool,
) -> Tuple[float, float, float, float]:
    y_train = labels_1d(y_train)
    y_test = labels_1d(y_test)
    try:
        train_pred = wrapper.predict(X_train)
        test_pred = wrapper.predict(X_test)
        train_acc = float(accuracy_score(y_train, train_pred))
        test_acc = float(accuracy_score(y_test, test_pred))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    train_auc = test_auc = np.nan
    try:
        train_proba = wrapper.predict_proba(X_train)
        test_proba = wrapper.predict_proba(X_test)
        if binary or (
            train_proba.ndim == 2 and train_proba.shape[1] == 2 and len(np.unique(y_train)) <= 2
        ):
            if train_proba.ndim == 1 or train_proba.shape[1] == 1:
                train_proba = np.stack([1 - train_proba.ravel(), train_proba.ravel()], axis=1)
            if test_proba.ndim == 1 or test_proba.shape[1] == 1:
                test_proba = np.stack([1 - test_proba.ravel(), test_proba.ravel()], axis=1)
            train_auc = float(roc_auc_score(y_train, train_proba[:, 1]))
            test_auc = float(roc_auc_score(y_test, test_proba[:, 1]))
        else:
            classes = np.unique(np.concatenate([y_train, y_test]))
            train_auc = _ovr_auc(y_train, train_proba, classes)
            test_auc = _ovr_auc(y_test, test_proba, classes)
    except Exception:
        pass
    return train_acc, test_acc, train_auc, test_auc


def _ovr_auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    scores = []
    for i, c in enumerate(classes):
        if i >= proba.shape[1]:
            break
        mask_pos = y_true == c
        if mask_pos.sum() == 0 or mask_pos.sum() == len(y_true):
            continue
        scores.append(roc_auc_score((y_true == c).astype(int), proba[:, i]))
    if not scores:
        return np.nan
    return float(np.mean(scores))


def interpretability_metrics(model: Any) -> Tuple[float, float]:
    ling = np.nan
    relax = np.nan
    if hasattr(model, "linguistic_richness"):
        try:
            ling = float(model.linguistic_richness())
        except Exception:
            ling = np.nan
    if hasattr(model, "relaxation_rate"):
        try:
            out = model.relaxation_rate()
            relax = float(out[0] if isinstance(out, tuple) else out)
        except Exception:
            relax = np.nan
    return ling, relax
