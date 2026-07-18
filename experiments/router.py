"""Central dispatcher for model-specific experiment protocols."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from experiments.protocols.admtsk import run_admtsk_protocol
from experiments.protocols.anfis import run_anfis_protocol
from experiments.protocols.common import ProtocolRunResult
from experiments.protocols.generic import run_generic_protocol
from experiments.protocols.unfis import run_unfis_protocol
from experiments.protocols.vsrp_anya import run_vsrp_anya_protocol
from experiments.registry import (
    PROTOCOL_ADMTSK,
    PROTOCOL_ANFIS,
    PROTOCOL_EFNN_NULLUNI,
    PROTOCOL_GENERIC,
    PROTOCOL_UNFIS,
    PROTOCOL_VSRP_ANYA,
    resolve_protocol_id,
)


def run_model_experiment(
    model_name: str,
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
    generic_runner: Optional[Callable[..., ProtocolRunResult]] = None,
) -> ProtocolRunResult:
    """Route a model to its paper protocol or the generic Evaluator path.

    EFNN-NullUni, VSRP-AnYa-EFS, ADMTSK, ANFIS, and UNFIS-c never use generic
    Adam/CE settings. All other models fall through to ``run_generic_protocol``
    (typically via ``generic_runner`` from ``Evaluator``).
    """
    protocol_id = resolve_protocol_id(model_name, model_factory)
    cfg = dict(config or {})
    common = dict(
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
        config=cfg,
    )

    if protocol_id == PROTOCOL_EFNN_NULLUNI:
        from experiments.protocols.efnn_nulluni import run_efnn_nulluni_protocol

        return run_efnn_nulluni_protocol(**common)

    if protocol_id == PROTOCOL_VSRP_ANYA:
        return run_vsrp_anya_protocol(**common)

    if protocol_id == PROTOCOL_ADMTSK:
        return run_admtsk_protocol(**common)

    if protocol_id == PROTOCOL_ANFIS:
        return run_anfis_protocol(**common)

    if protocol_id == PROTOCOL_UNFIS:
        return run_unfis_protocol(**common)

    return run_generic_protocol(**common, train_fn=generic_runner)
