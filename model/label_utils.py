"""Stable label encoding helpers for multi-class wrappers."""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np


def label_key(value: Any) -> Hashable:
    """Hash-stable key for class labels (including numpy scalars / strings)."""
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return int(value)
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def stable_unique(y: np.ndarray) -> np.ndarray:
    """Unique labels in first-appearance order (deterministic, not sorted)."""
    y = np.asarray(y).reshape(-1)
    seen = set()
    ordered = []
    for v in y.tolist():
        key = label_key(v)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(v)
    return np.asarray(ordered, dtype=object if any(isinstance(v, str) for v in ordered) else None)
