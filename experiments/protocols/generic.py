"""Generic Adam / CE / OneCycle protocol (non-paper models).

Paper models (EFNN-NullUni, VSRP-AnYa-EFS, ADMTSK) must never call this.
The shared ``Evaluator`` owns the full generic training loop; this module
exposes a thin entry point for the router when no Evaluator callback is
supplied.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from experiments.protocols.common import ProtocolRunResult


def run_generic_protocol(
    model_factory,
    wrapper_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    model_params: Optional[Dict[str, Any]] = None,
    binary: bool = False,
    device=None,
    seed: int = 0,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    train_fn: Optional[Callable[..., ProtocolRunResult]] = None,
) -> ProtocolRunResult:
    """Run the generic (non-paper) experiment path.

    Prefer passing ``train_fn`` from ``Evaluator`` (Adam / OneCycle / CE).
    Standalone calls without ``train_fn`` are not supported — use the
    Evaluator for classical neuro-fuzzy models.
    """
    if train_fn is not None:
        return train_fn(
            model_factory=model_factory,
            wrapper_factory=wrapper_factory,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_params=model_params,
            binary=binary,
            device=device,
            seed=seed,
            X_val=X_val,
            y_val=y_val,
            config=config,
        )
    raise NotImplementedError(
        "Generic protocol requires Evaluator.train_fn / _train_and_evaluate. "
        "Paper models should route via run_efnn_nulluni_protocol, "
        "run_vsrp_anya_protocol, or run_admtsk_protocol."
    )
