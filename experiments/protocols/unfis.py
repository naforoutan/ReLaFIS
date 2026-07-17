"""UNFIS-c paper protocol: KNN init + GqLM, rules=2, no Adam/CE path."""

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
from model.unfis import UNFIS


def run_unfis_protocol(
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
    """Train UNFIS-c with KNN initialization and GqLM (paper defaults).

    Ordinary benchmark defaults: ``rules=2``. Application-specific splits
    (Pima 50/25/25, WBC, DNA, etc.) are owned by the experiment object; this
    runner only consumes the provided train/test matrices.
    """
    del X_val, y_val
    cfg = dict(config or {})
    noise_std = float(cfg.get("noise_std", 0.0))
    apply_noise = cfg.get("apply_noise")

    params = dict(model_params or {})
    params["in_features"] = int(X_train.shape[1])
    params.setdefault("binary", binary)
    params.setdefault("rules", int(getattr(model_factory, "PAPER_RULE_COUNT", 2)))
    params.setdefault("drop_out_p", 0.0)
    params.setdefault("selection_epsilon", 1e-6)

    n_classes = int(len(np.unique(y_train)))
    if binary:
        params["out_features"] = 2
    else:
        params.setdefault("out_features", n_classes)

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
    wrapper = wrapper_factory(model, device=device)
    wrapper.fit(
        X_train,
        y_train,
        random_state=int(seed),
        minibatch_size=int(cfg.get("minibatch_size", 32)),
        lambda_=float(cfg.get("lambda_", cfg.get("gqlm_lambda", 1e3))),
        eta=float(cfg.get("eta", 1e-3)),
        beta=float(cfg.get("beta", 0.9)),
        max_iterations=int(cfg.get("max_iterations", cfg.get("epochs", 100))),
    )

    train_acc, test_acc, train_auc, test_auc = compute_acc_auc(
        wrapper, X_train, y_train, X_test_eval, y_test, binary=binary
    )
    ling, relax = interpretability_metrics(model)
    extras = {
        "model_name": "UNFIS-c",
        "implementation_status": "paper_aligned_reimplementation",
        "protocol": "knn_init_gqlm",
        "training_mode": "knn_initialization_gqlm",
        "rules": int(model.rules_count),
        "out_features": int(model.out_features),
        "selection_epsilon": float(model.selection_epsilon),
        "optimizer": "GqLM",
        "scheduler": None,
        "gqlm": dict(getattr(model, "training_metadata", {})),
        "init": dict(getattr(model, "init_metadata", {})),
        "adaptations": [
            "binary uses two class outputs with softmax/argmax",
            "GqLM uses autograd Jacobians (not gradient descent)",
        ],
    }
    return ProtocolRunResult(
        train_acc=train_acc,
        test_acc=test_acc,
        train_auc=train_auc,
        test_auc=test_auc,
        linguistic_richness=ling,
        relaxation_rate=relax,
        wrapper=wrapper,
        protocol_name="unfis",
        extras=extras,
    )
