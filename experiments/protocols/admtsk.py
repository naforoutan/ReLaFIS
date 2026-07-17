"""ADMTSK paper protocol: raw min-max [0,1], MSE eq. 9, fixed-LR Adam, R=3.

Leakage-safe train-only inner validation selects among the six paper
(learning_rate, batch_fraction) pairs. This nested selection detail is
documented as an implementation choice when the paper does not fully specify
an inner validation procedure — not as an exact original protocol statement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from experiments.protocols.common import (
    ProtocolRunResult,
    apply_optional_noise,
    compute_acc_auc,
    interpretability_metrics,
)
from model.admtsk import ADMTSK, SklearnADMTSKWrapper, train_admtsk_model
from model.label_utils import stable_unique


LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_FRACTIONS = [0.1, 0.2]


def _fit_minmax(
    X_train: np.ndarray,
    *others: Optional[np.ndarray],
) -> Tuple[MinMaxScaler, Tuple[np.ndarray, ...]]:
    """Fresh train-fitted MinMaxScaler to [0, 1]; never fit on others."""
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train = np.asarray(X_train, dtype=np.float64)
    Xt = scaler.fit_transform(X_train).astype(np.float32)
    out: List[np.ndarray] = [Xt]
    for X in others:
        if X is None:
            out.append(None)  # type: ignore[arg-type]
        else:
            out.append(scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32))
    return scaler, tuple(out)


def _build_paper_model(
    model_factory,
    *,
    in_features: int,
    out_features: int,
    binary: bool,
    model_params: Optional[Dict[str, Any]] = None,
    device=None,
) -> ADMTSK:
    incoming = dict(model_params or {})
    # Raise on conflicts — do not silently overwrite paper constraints.
    if "rules" in incoming and int(incoming["rules"]) != ADMTSK.PAPER_RULE_COUNT:
        raise ValueError(
            f"ADMTSK paper factory requires rules={ADMTSK.PAPER_RULE_COUNT}, "
            f"got {incoming['rules']}"
        )
    if "paper_init" in incoming and not bool(incoming["paper_init"]):
        raise ValueError("ADMTSK paper factory requires paper_init=True")
    if "adaptive" in incoming and not bool(incoming["adaptive"]):
        raise ValueError("ADMTSK paper factory requires adaptive=True")
    if "K" in incoming and abs(float(incoming["K"]) - 10.0) > 1e-12:
        raise ValueError(f"ADMTSK paper factory requires K=10, got {incoming['K']}")
    if "paper_mode" in incoming and not bool(incoming["paper_mode"]):
        raise ValueError("ADMTSK paper factory requires paper_mode=True")

    params = dict(incoming)
    params["in_features"] = int(in_features)
    params["out_features"] = int(out_features)
    params["binary"] = bool(binary)
    params["rules"] = int(ADMTSK.PAPER_RULE_COUNT)
    params["paper_init"] = True
    params["adaptive"] = True
    params["K"] = 10.0
    params["drop_out_p"] = float(params.get("drop_out_p", 0.0))
    params["paper_mode"] = True
    params.pop("max_rules", None)
    model = model_factory(**params, dtype=torch.float32)
    return model.to(device or torch.device("cpu"))


def select_admtsk_hyperparameters(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    binary: bool = False,
    model_params: Optional[Dict[str, Any]] = None,
    learning_rates: Sequence[float] = LEARNING_RATES,
    batch_fractions: Sequence[float] = BATCH_FRACTIONS,
    epochs: int = 50,
    n_inner_splits: int = 3,
    random_state: int = 0,
    device=None,
) -> Tuple[float, float, Dict[Tuple[float, float], float]]:
    """Train-only selection over the six paper (lr, batch_frac) pairs.

    Tie-break: higher val acc, then smaller learning rate, then smaller
    batch fraction.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).reshape(-1)
    classes = stable_unique(y_train)
    n_out = 2 if binary else int(len(classes))
    if binary and len(classes) != 2:
        raise ValueError("Binary hyperparameter selection needs exactly two classes.")

    min_count = int(np.min(np.bincount(np.unique(y_train, return_inverse=True)[1])))
    n_inner = int(min(n_inner_splits, min_count, max(X_train.shape[0] // 5, 2)))
    n_inner = max(n_inner, 2)
    if min_count < 2:
        raise ValueError("Cannot run inner validation: a class has fewer than 2 samples.")

    sss = StratifiedShuffleSplit(
        n_splits=n_inner,
        test_size=0.25,
        random_state=int(random_state),
    )
    scores: Dict[Tuple[float, float], List[float]] = {
        (float(lr), float(bf)): [] for lr in learning_rates for bf in batch_fractions
    }

    for lr in learning_rates:
        for bf in batch_fractions:
            for fold_i, (tr, va) in enumerate(sss.split(X_train, y_train)):
                scaler, (Xtr, Xva) = _fit_minmax(X_train[tr], X_train[va])
                del scaler
                model = _build_paper_model(
                    model_factory,
                    in_features=X_train.shape[1],
                    out_features=n_out,
                    binary=binary,
                    model_params=model_params,
                    device=device,
                )
                train_admtsk_model(
                    model,
                    Xtr,
                    y_train[tr],
                    learning_rate=float(lr),
                    batch_fraction=float(bf),
                    epochs=int(epochs),
                    random_state=int(random_state) + fold_i,
                    device=device,
                )
                wrap = SklearnADMTSKWrapper(model, device=device)
                pred = wrap.predict(Xva)
                scores[(float(lr), float(bf))].append(
                    float(accuracy_score(y_train[va], pred))
                )

    mean_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    best = sorted(
        mean_scores.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )[0][0]
    return best[0], best[1], mean_scores


def run_admtsk_single_split(
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
    epochs: int = 50,
    tune: bool = True,
    learning_rate: Optional[float] = None,
    batch_fraction: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ProtocolRunResult:
    """One outer split: optional HPO on train, final fit, test once."""
    cfg = dict(config or {})
    epochs = int(cfg.get("epochs", epochs))
    noise_std = float(cfg.get("noise_std", 0.0))
    apply_noise = cfg.get("apply_noise")
    device = device or torch.device("cpu")

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test).reshape(-1)

    classes = np.unique(y_train)
    n_out = 2 if binary else int(len(classes))
    if binary and n_out != 2:
        raise ValueError("Binary ADMTSK requires two classes.")

    candidate_scores: Dict[str, float] = {}
    if tune and learning_rate is None:
        lr, bf, mean_scores = select_admtsk_hyperparameters(
            model_factory,
            X_train,
            y_train,
            binary=binary,
            model_params=model_params,
            epochs=epochs,
            random_state=int(seed),
            device=device,
            n_inner_splits=int(cfg.get("inner_splits", 3)),
        )
        candidate_scores = {
            f"lr={k[0]}_bf={k[1]}": v for k, v in mean_scores.items()
        }
    else:
        lr = float(
            learning_rate
            if learning_rate is not None
            else cfg.get("lr", cfg.get("learning_rate", 0.001))
        )
        bf = float(
            batch_fraction
            if batch_fraction is not None
            else cfg.get("batch_frac", cfg.get("batch_fraction", 0.2))
        )

    scaler, (Xtr, Xte) = _fit_minmax(X_train, X_test)
    Xtr = apply_optional_noise(Xtr, noise_std, apply_noise)
    Xte = apply_optional_noise(Xte, noise_std, apply_noise)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = _build_paper_model(
        model_factory,
        in_features=X_train.shape[1],
        out_features=n_out,
        binary=binary,
        model_params=model_params,
        device=device,
    )
    train_admtsk_model(
        model,
        Xtr,
        y_train,
        learning_rate=lr,
        batch_fraction=bf,
        epochs=epochs,
        random_state=int(seed),
        device=device,
    )
    batch_size = int(model.training_metadata.get("batch_size", 1))

    wrapper = wrapper_factory(model, device=device)
    train_acc, test_acc, train_auc, test_auc = compute_acc_auc(
        wrapper, Xtr, y_train, Xte, y_test, binary=binary
    )
    ling, relax = interpretability_metrics(model)

    extras = {
        "model_name": "ADMTSK",
        "implementation_status": "paper_aligned_reimplementation",
        "protocol": "single_outer_split_with_train_only_hpo",
        "preprocessing": "raw_train_fitted_minmax_0_1",
        "scaler": type(scaler).__name__,
        "rules": int(model.rules_count),
        "epochs": epochs,
        "loss": "paper_one_hot_mse",
        "optimizer": "fixed_lr_adam",
        "scheduler": None,
        "selected_learning_rate": lr,
        "selected_batch_fraction": bf,
        "batch_size": batch_size,
        "hyperparameter_grid": {
            "learning_rate": list(LEARNING_RATES),
            "batch_fraction": list(BATCH_FRACTIONS),
        },
        "grid_validation_scores": candidate_scores,
        "lambda_unclamped": float(model.lambda_unclamped),
        "lambda_value": float(model.lambda_value),
        "lambda_was_clamped": bool(model.lambda_was_clamped),
        "n_classes": int(n_out),
        "adaptations": [
            "adaptive lambda uses the paper equation without a minimum-1 clamp",
            "train-only inner validation used for leakage-safe hyperparameter selection",
            "softmax probabilities are uncalibrated compatibility scores",
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
        protocol_name="admtsk",
        extras=extras,
    )


def run_admtsk_repeated_cv(
    model_factory,
    wrapper_factory,
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_params: Optional[Dict[str, Any]] = None,
    binary: bool = False,
    device=None,
    seed: int = 0,
    epochs: int = 50,
    n_splits: int = 10,
    n_repeats: int = 5,
    config: Optional[Dict[str, Any]] = None,
) -> ProtocolRunResult:
    """RepeatedStratifiedKFold(10, 5) → 50 outer scores (when stratification allows)."""
    cfg = dict(config or {})
    epochs = int(cfg.get("epochs", epochs))
    device = device or torch.device("cpu")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)

    rskf = RepeatedStratifiedKFold(
        n_splits=int(n_splits),
        n_repeats=int(n_repeats),
        random_state=int(seed),
    )
    fold_scores: List[float] = []
    fold_metadata: List[Dict[str, Any]] = []
    last_wrapper = None

    for fold_id, (tr, te) in enumerate(rskf.split(X, y)):
        repeat_index = fold_id // n_splits
        fold_index = fold_id % n_splits
        result = run_admtsk_single_split(
            model_factory,
            wrapper_factory,
            X[tr],
            y[tr],
            X[te],
            y[te],
            model_params=model_params,
            binary=binary,
            device=device,
            seed=int(seed) + fold_id,
            epochs=epochs,
            tune=bool(cfg.get("tune", True)),
            config=cfg,
        )
        fold_scores.append(float(result.test_acc))
        last_wrapper = result.wrapper
        fold_metadata.append(
            {
                "repeat_index": int(repeat_index),
                "fold_index": int(fold_index),
                "seed": int(seed) + fold_id,
                "selected_learning_rate": result.extras.get("selected_learning_rate"),
                "selected_batch_fraction": result.extras.get("selected_batch_fraction"),
                "batch_size": result.extras.get("batch_size"),
                "n_classes": result.extras.get("n_classes"),
                "lambda_unclamped": result.extras.get("lambda_unclamped"),
                "lambda_value": result.extras.get("lambda_value"),
                "lambda_was_clamped": result.extras.get("lambda_was_clamped"),
                "rules": result.extras.get("rules"),
                "test_accuracy": float(result.test_acc),
            }
        )

    scores_arr = np.asarray(fold_scores, dtype=np.float64)
    mean_acc = float(np.mean(scores_arr))
    std_acc = float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0

    extras = {
        "model_name": "ADMTSK",
        "implementation_status": "paper_aligned_reimplementation",
        "protocol": "repeated_stratified_10_fold_5_repeats",
        "preprocessing": "raw_train_fitted_minmax_0_1",
        "rules": 3,
        "epochs": epochs,
        "loss": "paper_one_hot_mse",
        "optimizer": "fixed_lr_adam",
        "scheduler": None,
        "hyperparameter_grid": {
            "learning_rate": list(LEARNING_RATES),
            "batch_fraction": list(BATCH_FRACTIONS),
        },
        "fold_scores": fold_scores,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "std_convention": "np.std(ddof=1)",
        "n_outer_scores": len(fold_scores),
        "fold_metadata": fold_metadata,
        "adaptations": [
            "adaptive lambda uses the paper equation without a minimum-1 clamp",
            "train-only inner validation used for leakage-safe hyperparameter selection",
            "softmax probabilities are uncalibrated compatibility scores",
        ],
    }

    # Diagnostic train metrics on last fold wrapper (not paper summary).
    return ProtocolRunResult(
        train_acc=float("nan"),
        test_acc=mean_acc,
        train_auc=float("nan"),
        test_auc=float("nan"),
        linguistic_richness=float("nan"),
        relaxation_rate=float("nan"),
        wrapper=last_wrapper,
        protocol_name="admtsk",
        extras=extras,
    )


def run_admtsk_protocol(
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
    """ADMTSK paper router entry (Evaluator / ``run_model_experiment``).

    Modes
    -----
    * ``repeated_stratified_cv=True`` (paper comparison): concatenate the
      provided split into the full matrix and run 10×5 RSKF.
    * otherwise: single outer split with train-only HPO (or fixed lr/bf).
    """
    del X_val, y_val
    cfg = dict(config or {})
    epochs = int(cfg.get("epochs", ADMTSK.PAPER_EPOCH_COUNT))

    # Fast path for unit tests / ablations.
    tune = bool(cfg.get("tune", cfg.get("tune_hyperparameters", True)))
    if cfg.get("epochs") is not None and int(cfg["epochs"]) < ADMTSK.PAPER_EPOCH_COUNT:
        # Short smoke runs still use the dedicated trainer, not generic CE.
        pass

    if bool(cfg.get("repeated_stratified_cv", False)):
        X = np.vstack(
            [np.asarray(X_train, dtype=np.float64), np.asarray(X_test, dtype=np.float64)]
        )
        y = np.concatenate(
            [np.asarray(y_train).reshape(-1), np.asarray(y_test).reshape(-1)]
        )
        return run_admtsk_repeated_cv(
            model_factory,
            wrapper_factory,
            X,
            y,
            model_params=model_params,
            binary=binary,
            device=device,
            seed=seed,
            epochs=epochs,
            n_splits=int(cfg.get("n_splits", 10)),
            n_repeats=int(cfg.get("n_repeats", 5)),
            config=cfg,
        )

    # When caller fixes lr/batch without asking for grid search.
    fixed = cfg.get("lr") is not None or cfg.get("learning_rate") is not None
    if fixed and not tune:
        return run_admtsk_single_split(
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
            epochs=epochs,
            tune=False,
            config=cfg,
        )

    # Default Evaluator path: HPO on the provided outer-train split.
    # Smoke configs may set tune=False and short epochs.
    if "tune" not in cfg and "tune_hyperparameters" not in cfg:
        # If epochs are abbreviated (tests), skip expensive 6-config HPO.
        tune = epochs >= ADMTSK.PAPER_EPOCH_COUNT

    return run_admtsk_single_split(
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
        epochs=epochs,
        tune=tune,
        config=cfg,
    )
