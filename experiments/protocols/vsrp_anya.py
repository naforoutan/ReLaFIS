"""VSRP-AnYa-EFS paper protocol: [-1,1] MinMax, online fit, frozen test.

Compression profiles (paper):
  standard:      k ∈ {2, 3, 4},      10-fold CV
  advertisement: k ∈ {5, 10, 15},    10-fold CV
  gisette:       k ∈ {10,20,30,40,50}, 5-fold CV

Mode A — paper-table candidate reporting: run every k independently.
Mode B — one selected baseline: inner CV on outer-train only.

Train-only MinMaxScaler(-1,1); no test-time model updates.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from experiments.protocols.common import (
    ProtocolRunResult,
    apply_optional_noise,
    compute_acc_auc,
    interpretability_metrics,
)


STANDARD_COMPRESSION_FACTORS = (2, 3, 4)
ADVERTISEMENT_COMPRESSION_FACTORS = (5, 10, 15)
GISETTE_COMPRESSION_FACTORS = (10, 20, 30, 40, 50)

DATASET_PROFILES = {
    "standard": {
        "compression_factors": STANDARD_COMPRESSION_FACTORS,
        "n_splits": 10,
    },
    "advertisement": {
        "compression_factors": ADVERTISEMENT_COMPRESSION_FACTORS,
        "n_splits": 10,
    },
    "gisette": {
        "compression_factors": GISETTE_COMPRESSION_FACTORS,
        "n_splits": 5,
    },
}


def _fit_minmax_m1_1(
    X_train: np.ndarray,
    *others: Optional[np.ndarray],
) -> Tuple[MinMaxScaler, Tuple]:
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_train = np.asarray(X_train, dtype=np.float64)
    Xt = scaler.fit_transform(X_train).astype(np.float32)
    out = [Xt]
    for X in others:
        if X is None:
            out.append(None)
        else:
            out.append(
                scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
            )
    return scaler, tuple(out)


def _preserve_order_indices(indices: np.ndarray) -> np.ndarray:
    """Keep original within-fold order (sorted original indices)."""
    return np.sort(np.asarray(indices, dtype=np.int64))


def select_compression_ratio(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    candidates: Sequence[int] = STANDARD_COMPRESSION_FACTORS,
    binary: bool = False,
    model_params: Optional[Dict[str, Any]] = None,
    n_inner_splits: int = 3,
    random_state: int = 0,
    device=None,
) -> Tuple[int, Dict[int, float]]:
    """Train-only inner selection of compression factor k.

    Tie-break (documented):
      1. higher mean validation accuracy
      2. larger k (smaller projected dim) when accuracy ties within 1e-12
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).reshape(-1)
    classes = np.unique(y_train)
    n_out = 2 if binary else int(len(classes))
    if binary and len(classes) != 2:
        raise ValueError("Binary selection needs exactly two classes.")

    counts = np.bincount(np.unique(y_train, return_inverse=True)[1])
    min_count = int(counts.min()) if len(counts) else 0
    if min_count < 2:
        raise ValueError("Inner CV impossible: a class has fewer than 2 samples.")
    n_inner = max(2, min(int(n_inner_splits), min_count, X_train.shape[0] // 4))

    sss = StratifiedShuffleSplit(
        n_splits=n_inner, test_size=0.25, random_state=int(random_state)
    )
    # Materialize identical splits for every candidate.
    splits = list(sss.split(X_train, y_train))
    scores: Dict[int, List[float]] = {int(k): [] for k in candidates}
    device = device or torch.device("cpu")

    for k in candidates:
        k = int(k)
        for fold_i, (tr, va) in enumerate(splits):
            tr = _preserve_order_indices(tr)
            va = _preserve_order_indices(va)
            scaler, (Xtr, Xva) = _fit_minmax_m1_1(X_train[tr], X_train[va])
            del scaler
            params = dict(model_params or {})
            params.update(
                {
                    "in_features": int(X_train.shape[1]),
                    "out_features": n_out,
                    "binary": binary,
                    "rules": 1,
                    "drop_out_p": 0.0,
                    "max_rules": None,
                    "compression_ratio": k,
                    "proj_dim": None,
                    "seed": int(random_state) + 1000 * k + fold_i,
                    "p_init": 500.0,
                    "projection_mode": "paper_dynamic",
                }
            )
            model = model_factory(**params, dtype=torch.float32).to(device)
            model.fit(Xtr, y_train[tr])
            from model.vsrp_anya import SklearnVSRPAnyaEFSWrapper

            wrap = SklearnVSRPAnyaEFSWrapper(model, device=device)
            pred = wrap.predict(Xva)
            scores[k].append(float(accuracy_score(y_train[va], pred)))

    mean_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    # Higher acc, then larger k (more compression) on ties.
    best = sorted(mean_scores.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]
    return int(best), mean_scores


def run_vsrp_single_split(
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
    compression_ratio: Optional[int] = None,
    tune_compression: bool = False,
    candidates: Sequence[int] = STANDARD_COMPRESSION_FACTORS,
    config: Optional[Dict[str, Any]] = None,
) -> ProtocolRunResult:
    """Fit on train (sequential), predict test with frozen model."""
    cfg = dict(config or {})
    noise_std = float(cfg.get("noise_std", 0.0))
    apply_noise = cfg.get("apply_noise")
    device = device or torch.device("cpu")

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test).reshape(-1)

    candidate_scores: Dict[int, float] = {}
    if tune_compression and compression_ratio is None:
        selected_k, candidate_scores = select_compression_ratio(
            model_factory,
            X_train,
            y_train,
            candidates=candidates,
            binary=binary,
            model_params=model_params,
            random_state=int(seed),
            device=device,
            n_inner_splits=int(cfg.get("inner_splits", 3)),
        )
    else:
        selected_k = int(
            compression_ratio
            if compression_ratio is not None
            else cfg.get("compression_ratio", (model_params or {}).get("compression_ratio", 3))
        )

    scaler, (Xtr, Xte) = _fit_minmax_m1_1(X_train, X_test)
    Xtr = apply_optional_noise(Xtr, noise_std, apply_noise)
    Xte = apply_optional_noise(Xte, noise_std, apply_noise)

    n_out = 2 if binary else int(len(np.unique(y_train)))
    params = dict(model_params or {})
    params.update(
        {
            "in_features": int(X_train.shape[1]),
            "out_features": n_out,
            "binary": binary,
            "rules": int(params.get("rules", 1)),
            "drop_out_p": float(params.get("drop_out_p", 0.0)),
            "max_rules": params.get("max_rules", None),
            "compression_ratio": selected_k,
            "proj_dim": None,
            "seed": int(seed),
            "p_init": float(params.get("p_init", 500.0)),
            "projection_mode": params.get("projection_mode", "paper_dynamic"),
        }
    )

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = model_factory(**params, dtype=torch.float32).to(device)
    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    train_time = float(time.perf_counter() - t0)
    initial_rules = 1  # first sample always creates one cloud
    final_rules = int(model.rules_count)

    # Freeze: snapshot before test prediction.
    snap_before = model.snapshot_state()
    wrapper = wrapper_factory(model, device=device)
    train_acc, test_acc, train_auc, test_auc = compute_acc_auc(
        wrapper, Xtr, y_train, Xte, y_test, binary=binary
    )
    snap_after = model.snapshot_state()
    test_time_updates = 0
    # Lightweight equality check on scalars / shapes.
    frozen_ok = (
        snap_before["rules_count"] == snap_after["rules_count"]
        and snap_before["n_seen"] == snap_after["n_seen"]
        and snap_before["next_rule_index"] == snap_after["next_rule_index"]
        and np.allclose(snap_before["centers"], snap_after["centers"])
        and np.allclose(snap_before["Q"], snap_after["Q"])
    )
    if not frozen_ok:
        raise RuntimeError("Test prediction mutated VSRP-AnYa-EFS state.")

    ling, relax = interpretability_metrics(model)
    extras = {
        "model_name": "VSRP-AnYa-EFS",
        "implementation_status": "paper_aligned_reimplementation",
        "protocol": "train_fit_frozen_test",
        "preprocessing": "raw_train_fitted_minmax_m1_1",
        "scaler": type(scaler).__name__,
        "feature_range": (-1.0, 1.0),
        "compression_factor": selected_k,
        "compression_ratio": selected_k,
        "projection_mode": str(getattr(model, "projection_mode", "paper_dynamic")),
        "evolved_rule_count": final_rules,
        "projected_dimension": int(model.proj_dim),
        "extended_input_dimension": int(model.d_ext),
        "grid_validation_scores": candidate_scores,
        "initial_rules": initial_rules,
        "final_rules": final_rules,
        "rules_count": final_rules,
        "training_time_sec": train_time,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "test_time_update_count": test_time_updates,
        "test_time_frozen": True,
        "seed": int(seed),
        "adaptations": [
            "one output column per class (binary uses two outputs + one-hot + argmax)",
            "sklearn probability-like compatibility scores",
            "train-only preprocessing and hyperparameter selection",
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
        protocol_name="vsrp_anya",
        extras=extras,
    )


def run_vsrp_stratified_cv(
    model_factory,
    wrapper_factory,
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_params: Optional[Dict[str, Any]] = None,
    binary: bool = False,
    device=None,
    seed: int = 0,
    dataset_profile: str = "standard",
    mode: str = "leakage_safe_selection_mode",  # or paper_table_mode
    config: Optional[Dict[str, Any]] = None,
) -> ProtocolRunResult:
    """Stratified K-fold paper protocol for VSRP-AnYa-EFS."""
    cfg = dict(config or {})
    if dataset_profile not in DATASET_PROFILES:
        raise ValueError(
            f"Unknown dataset_profile {dataset_profile!r}; "
            f"expected one of {sorted(DATASET_PROFILES)}"
        )
    profile = DATASET_PROFILES[dataset_profile]
    candidates = tuple(
        cfg.get("compression_factors", profile["compression_factors"])
    )
    n_splits = int(cfg.get("n_splits", profile["n_splits"]))

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=int(seed)
    )

    if mode in ("per_candidate", "paper_table_mode"):
        # Mode A: one mean±std per k (store nested results).
        per_k: Dict[int, List[float]] = {int(k): [] for k in candidates}
        fold_metadata: List[Dict[str, Any]] = []
        last_wrapper = None
        for fold_i, (tr, te) in enumerate(skf.split(X, y)):
            tr = _preserve_order_indices(tr)
            te = _preserve_order_indices(te)
            for k in candidates:
                result = run_vsrp_single_split(
                    model_factory,
                    wrapper_factory,
                    X[tr],
                    y[tr],
                    X[te],
                    y[te],
                    model_params=model_params,
                    binary=binary,
                    device=device,
                    seed=int(seed) + 100 * fold_i + int(k),
                    compression_ratio=int(k),
                    tune_compression=False,
                    config=cfg,
                )
                per_k[int(k)].append(float(result.test_acc))
                last_wrapper = result.wrapper
                fold_metadata.append(
                    {
                        "fold_index": fold_i,
                        "compression_factor": int(k),
                        "test_accuracy": float(result.test_acc),
                        "final_rules": result.extras.get("final_rules"),
                        "seed": int(seed) + 100 * fold_i + int(k),
                    }
                )
        summary = {
            int(k): {
                "mean_accuracy": float(np.mean(v)),
                "std_accuracy": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "fold_scores": v,
            }
            for k, v in per_k.items()
        }
        # Use best mean k as headline test_acc (reporting only; not selection by test).
        best_k = max(summary.items(), key=lambda kv: kv[1]["mean_accuracy"])[0]
        return ProtocolRunResult(
            train_acc=float("nan"),
            test_acc=summary[best_k]["mean_accuracy"],
            train_auc=float("nan"),
            test_auc=float("nan"),
            linguistic_richness=float("nan"),
            relaxation_rate=float("nan"),
            wrapper=last_wrapper,
            protocol_name="vsrp_anya",
            extras={
                "model_name": "VSRP-AnYa-EFS",
                "implementation_status": "paper_aligned_reimplementation",
                "protocol": f"stratified_{n_splits}_fold_per_candidate",
                "dataset_profile": dataset_profile,
                "preprocessing": "raw_train_fitted_minmax_m1_1",
                "mode": "per_candidate",
                "per_compression_results": summary,
                "fold_metadata": fold_metadata,
                "std_convention": "np.std(ddof=1)",
                "note": (
                    "Headline test_acc is the best mean over k for convenience; "
                    "paper tables should report each k separately from "
                    "per_compression_results."
                ),
            },
        )

    if mode not in ("selected", "leakage_safe_selection_mode"):
        raise ValueError(
            f"Unknown mode {mode!r}; expected paper_table_mode / "
            "leakage_safe_selection_mode (or legacy per_candidate / selected)."
        )

    # Mode B: select k on outer-train, evaluate once on outer-test.
    fold_scores: List[float] = []
    fold_metadata = []
    last_wrapper = None
    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        tr = _preserve_order_indices(tr)
        te = _preserve_order_indices(te)
        result = run_vsrp_single_split(
            model_factory,
            wrapper_factory,
            X[tr],
            y[tr],
            X[te],
            y[te],
            model_params=model_params,
            binary=binary,
            device=device,
            seed=int(seed) + fold_i,
            tune_compression=True,
            candidates=candidates,
            config=cfg,
        )
        fold_scores.append(float(result.test_acc))
        last_wrapper = result.wrapper
        fold_metadata.append(
            {
                "fold_index": fold_i,
                "seed": int(seed) + fold_i,
                "selected_compression_factor": result.extras.get("compression_factor"),
                "projected_dimension": result.extras.get("projected_dimension"),
                "final_rules": result.extras.get("final_rules"),
                "training_time_sec": result.extras.get("training_time_sec"),
                "test_accuracy": float(result.test_acc),
                "test_time_update_count": 0,
                "grid_validation_scores": result.extras.get("grid_validation_scores"),
            }
        )

    scores_arr = np.asarray(fold_scores, dtype=np.float64)
    return ProtocolRunResult(
        train_acc=float("nan"),
        test_acc=float(np.mean(scores_arr)),
        train_auc=float("nan"),
        test_auc=float("nan"),
        linguistic_richness=float("nan"),
        relaxation_rate=float("nan"),
        wrapper=last_wrapper,
        protocol_name="vsrp_anya",
        extras={
            "model_name": "VSRP-AnYa-EFS",
            "implementation_status": "paper_aligned_reimplementation",
            "protocol": f"stratified_{n_splits}_fold_selected_k",
            "dataset_profile": dataset_profile,
            "preprocessing": "raw_train_fitted_minmax_m1_1",
            "mode": "selected",
            "compression_candidates": list(candidates),
            "fold_scores": fold_scores,
            "mean_accuracy": float(np.mean(scores_arr)),
            "std_accuracy": float(np.std(scores_arr, ddof=1))
            if len(scores_arr) > 1
            else 0.0,
            "std_convention": "np.std(ddof=1)",
            "n_outer_scores": len(fold_scores),
            "fold_metadata": fold_metadata,
            "adaptations": [
                "one output column per class (binary uses two outputs + one-hot + argmax)",
                "sklearn probability-like compatibility scores",
                "train-only preprocessing and hyperparameter selection",
            ],
        },
    )


def run_vsrp_anya_protocol(
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
    """Evaluator / router entry for VSRP-AnYa-EFS."""
    del X_val, y_val
    cfg = dict(config or {})

    if bool(cfg.get("stratified_cv", False)):
        X = np.vstack(
            [np.asarray(X_train, dtype=np.float64), np.asarray(X_test, dtype=np.float64)]
        )
        y = np.concatenate(
            [np.asarray(y_train).reshape(-1), np.asarray(y_test).reshape(-1)]
        )
        return run_vsrp_stratified_cv(
            model_factory,
            wrapper_factory,
            X,
            y,
            model_params=model_params,
            binary=binary,
            device=device,
            seed=seed,
            dataset_profile=str(cfg.get("dataset_profile", "standard")),
            mode=str(cfg.get("cv_mode", "leakage_safe_selection_mode")),
            config=cfg,
        )

    tune = bool(cfg.get("tune_compression", False))
    profile = str(cfg.get("dataset_profile", "standard"))
    candidates = DATASET_PROFILES.get(profile, DATASET_PROFILES["standard"])[
        "compression_factors"
    ]
    if "compression_factors" in cfg:
        candidates = tuple(cfg["compression_factors"])

    return run_vsrp_single_split(
        model_factory,
        wrapper_factory,
        X_train,
        y_train,
        X_test,
        y_test,
        model_params=model_params,
        binary=binary,
        device=device,
        seed=seed,
        compression_ratio=cfg.get("compression_ratio"),
        tune_compression=tune,
        candidates=candidates,
        config=cfg,
    )
