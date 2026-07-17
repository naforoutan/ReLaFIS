"""ANFIS paper protocol: Jang hybrid LSE consequents + GD antecedents."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from experiments.protocols.common import (
    ProtocolRunResult,
    apply_optional_noise,
    compute_acc_auc,
    interpretability_metrics,
)


def run_anfis_protocol(
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
) -> ProtocolRunResult:
    """Train ANFIS with hybrid LSE+GD — never Adam / CE / OneCycle / recon."""
    del X_val, y_val
    cfg = dict(config or {})
    epochs = int(cfg.get("epochs", 50))
    lr = float(cfg.get("lr", 1e-2))
    batch_size = cfg.get("batch_size", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    noise_std = float(cfg.get("noise_std", 0.0))
    apply_noise = cfg.get("apply_noise")

    params = dict(model_params or {})
    params["in_features"] = int(X_train.shape[1])
    params.setdefault("binary", binary)
    if binary:
        params["out_features"] = 1
    else:
        params.setdefault("out_features", int(len(np.unique(y_train))))
    params.setdefault("rules", 3)
    params.setdefault("drop_out_p", 0.0)
    params.pop("max_rules", None)

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test).reshape(-1)

    X_train = apply_optional_noise(X_train, noise_std, apply_noise)
    X_test_eval = apply_optional_noise(X_test, noise_std, apply_noise)

    device = device or torch.device("cpu")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = model_factory(**params, dtype=torch.float32).to(device)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    wrapper = wrapper_factory(model, device=device)
    train_acc, test_acc, train_auc, test_auc = compute_acc_auc(
        wrapper, X_train, y_train, X_test_eval, y_test, binary=binary
    )
    ling, relax = interpretability_metrics(model)
    return ProtocolRunResult(
        train_acc=train_acc,
        test_acc=test_acc,
        train_auc=train_auc,
        test_auc=test_auc,
        linguistic_richness=ling,
        relaxation_rate=relax,
        wrapper=wrapper,
        protocol_name="anfis",
        extras={
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "training_mode": "hybrid_lse_gradient",
            "loss": "mse",
        },
    )
