"""EFNN-NullUni paper protocol: train-only grid CV + prequential evolve."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments.protocols.common import (
    ProtocolRunResult,
    apply_optional_noise,
    interpretability_metrics,
)
from model.efnn_nulluni.soda import SODA_GRID_SIZE_CANDIDATES


def paper_auc_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Paper-style binary AUC: ``0.5 * (sensitivity + specificity)``."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    classes = np.unique(y_true)
    if len(classes) != 2:
        return float("nan")
    neg, pos = classes[0], classes[1]
    tp = np.sum((y_true == pos) & (y_pred == pos))
    tn = np.sum((y_true == neg) & (y_pred == neg))
    fp = np.sum((y_true == neg) & (y_pred == pos))
    fn = np.sum((y_true == pos) & (y_pred == neg))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return float(0.5 * (sens + spec))


def select_efnn_grid_size(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_params: Optional[Dict[str, Any]] = None,
    candidates: Sequence[int] = (2, 3, 4, 5, 6, 7),
    n_splits: int = 5,
    random_state: int = 0,
    binary: bool = False,
    device=None,
) -> Tuple[int, Dict[int, float]]:
    """Train-only stratified CV over SODA ``grid_size`` candidates.

    Never reads an evolving/test stream. Tie-break: smaller grid_size wins.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).reshape(-1)
    cand = [int(c) for c in candidates]
    for c in cand:
        if c not in SODA_GRID_SIZE_CANDIDATES:
            raise ValueError(f"Invalid grid_size candidate {c}")

    classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    if min_count < 2 or X_train.shape[0] < 4:
        raise ValueError(
            "Train-only grid_size selection is impossible: "
            "need at least two samples per class."
        )
    n_splits = int(min(n_splits, min_count, X_train.shape[0] // 2))
    n_splits = max(n_splits, 2)

    params_base = dict(model_params or {})
    params_base["in_features"] = int(X_train.shape[1])
    params_base.setdefault("binary", binary)
    if binary:
        params_base["out_features"] = 1
    else:
        params_base.setdefault("out_features", int(len(classes)))
    params_base.setdefault("rules", 1)
    params_base.setdefault("drop_out_p", 0.0)
    params_base.setdefault("max_rules", None)
    params_base.pop("grid_size", None)

    device = device or torch.device("cpu")
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=int(random_state)
    )
    scores: Dict[int, List[float]] = {c: [] for c in cand}

    for grid in cand:
        for fold_i, (tr, va) in enumerate(skf.split(X_train, y_train)):
            model = model_factory(
                **params_base, grid_size=grid, dtype=torch.float32
            ).to(device)
            model.fit(X_train[tr], y_train[tr])
            # Fresh model per candidate/fold — score fold validation only.
            from model.EFNN_NullUni import SklearnEFNNWrapper

            wrap = SklearnEFNNWrapper(model, device=device)
            pred = wrap.predict(X_train[va])
            scores[grid].append(float(accuracy_score(y_train[va], pred)))

    mean_scores = {c: float(np.mean(v)) for c, v in scores.items()}
    # Highest mean; ties → smaller grid_size.
    best = sorted(
        mean_scores.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )[0][0]
    return int(best), mean_scores


def _prequential_stream(
    wrapper,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    *,
    event_log: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[float]]:
    """Predict-then-update on each stream sample (never update-then-predict)."""
    predictions = []
    positive_scores: List[float] = []
    for i in range(X_stream.shape[0]):
        x_one = X_stream[i].reshape(1, -1)
        y_one = np.asarray([y_stream[i]])
        if event_log is not None:
            event_log.append("predict")
        pred_i = wrapper.predict(x_one)[0]
        predictions.append(pred_i)
        if hasattr(wrapper, "predict_proba"):
            proba_i = wrapper.predict_proba(x_one)
            if proba_i.ndim == 2 and proba_i.shape[1] == 2:
                positive_scores.append(float(proba_i[0, 1]))
        if event_log is not None:
            event_log.append("partial_fit")
        wrapper.partial_fit(x_one, y_one)
    return np.asarray(predictions), positive_scores


def run_efnn_nulluni_protocol(
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
    """EFNN paper-comparison protocol (controlled reimplementation).

    Uses Evaluator ``X_train`` as the initial 70% fit set and ``X_test`` as
    the ordered evolving 30% stream (predict-then-``partial_fit``).

    Train-only ``grid_size`` selection over ``{2..7}`` unless fixed in config.
    """
    del X_val, y_val
    cfg = dict(config or {})
    noise_std = float(cfg.get("noise_std", 0.0))
    apply_noise = cfg.get("apply_noise")
    fixed_grid = cfg.get("grid_size", None)
    tune_grid = bool(cfg.get("tune_grid_size", True))
    n_splits = int(cfg.get("grid_cv_splits", 5))
    event_log: List[str] = []

    params = dict(model_params or {})
    params["in_features"] = int(X_train.shape[1])
    params.setdefault("binary", binary)
    if binary:
        params["out_features"] = 1
    else:
        params.setdefault("out_features", int(len(np.unique(y_train))))
    params.setdefault("rules", params.get("rules", 1))
    params.setdefault("drop_out_p", 0.0)
    params.setdefault("max_rules", cfg.get("max_rules", None))

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test).reshape(-1)

    X_train = apply_optional_noise(X_train, noise_std, apply_noise)
    X_stream = apply_optional_noise(X_test, noise_std, apply_noise)
    y_stream = y_test

    device = device or torch.device("cpu")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    candidate_scores: Dict[int, float] = {}
    grid_bypassed = False
    if not tune_grid:
        selected_grid = int(
            fixed_grid
            if fixed_grid is not None
            else params.get("grid_size", 5)
        )
        grid_bypassed = True
    else:
        try:
            selected_grid, candidate_scores = select_efnn_grid_size(
                model_factory,
                X_train,
                y_train,
                model_params=params,
                candidates=tuple(sorted(SODA_GRID_SIZE_CANDIDATES)),
                n_splits=n_splits,
                random_state=int(seed),
                binary=binary,
                device=device,
            )
        except ValueError:
            # Tiny training sets: fall back to mid-range default.
            selected_grid = int(
                fixed_grid
                if fixed_grid is not None
                else params.get("grid_size", 5)
            )
            grid_bypassed = True
            candidate_scores = {}

    params["grid_size"] = int(selected_grid)
    model = model_factory(**params, dtype=torch.float32).to(device)
    model.fit(X_train, y_train)
    initial_rules = int(model.rules_count)

    wrapper = wrapper_factory(model, device=device)
    # Sync wrapper fitted attrs after model.fit
    if hasattr(wrapper, "classes_"):
        wrapper.classes_ = np.asarray(model._classes)
        wrapper.is_fitted_ = True

    stream_pred, pos_scores = _prequential_stream(
        wrapper, X_stream, y_stream, event_log=event_log
    )
    final_rules = int(model.rules_count)

    stream_acc = float(accuracy_score(y_stream, stream_pred))
    # Also report train accuracy after learning the stream (diagnostic only).
    train_pred = wrapper.predict(X_train)
    train_acc = float(accuracy_score(y_train, train_pred))

    paper_auc = float("nan")
    roc_auc = float("nan")
    if binary and len(np.unique(y_stream)) == 2:
        paper_auc = paper_auc_binary(y_stream, stream_pred)
        try:
            if pos_scores:
                roc_auc = float(roc_auc_score(y_stream, np.asarray(pos_scores)))
        except Exception:
            roc_auc = float("nan")

    # Prequential order confirmation: predict before every partial_fit.
    prequential_ok = event_log[0::2] == ["predict"] * (len(event_log) // 2) and (
        event_log[1::2] == ["partial_fit"] * (len(event_log) // 2)
    )

    ling, relax = interpretability_metrics(model)
    extras = {
        "model_name": "EFNN-NullUni",
        "protocol": "70_30_prequential_30_repetitions",
        "implementation_status": "controlled_reimplementation",
        "evolution_mode": getattr(
            model, "evolution_mode", "approximate_incremental_soda"
        ),
        "seed": int(seed),
        "selected_grid_size": int(selected_grid),
        "grid_validation_scores": candidate_scores,
        "grid_tuning_bypassed": grid_bypassed,
        "initial_rules": initial_rules,
        "final_rules": final_rules,
        "accuracy": stream_acc,
        "paper_auc": paper_auc,
        "roc_auc": roc_auc,
        "n_stream_samples": int(X_stream.shape[0]),
        "prequential_order_ok": bool(prequential_ok),
        "recommended_n_runs": 30,
        "preprocessing": "evaluator_split_only; fit transform on train portion only",
        "adaptations": [
            "approximate incremental SODA update",
            "LOFO operator-orientation complement",
        ],
        "connector_note": (
            "connector parameters are identical by default; "
            "connector diversity remains zero unless explicitly configured or learned"
        ),
    }

    return ProtocolRunResult(
        train_acc=train_acc,
        test_acc=stream_acc,
        train_auc=float("nan"),
        test_auc=paper_auc if binary else roc_auc,
        linguistic_richness=ling,
        relaxation_rate=relax,
        wrapper=wrapper,
        protocol_name="efnn_nulluni",
        extras=extras,
    )
