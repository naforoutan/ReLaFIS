"""
An independent paper-aligned PyTorch reproduction of UNFIS-c.

Reference
---------
A. Salimi-Badr, "UNFIS: A Novel Neuro-Fuzzy Inference System with
Unstructured Fuzzy Rules," Neurocomputing 579 (2024) 127437.

This is not official author code. Exact numeric reproduction of published
tables is not claimed until datasets, splits, and GqLM details are matched.

Architecture (UNFIS-c)
----------------------
Eq. (6)  ζ = sigmoid(s)          ζ→1 selected, ζ→0 don't-care
Eq. (14) μ = Gaussian(x; m, σ)
Eq. (8)  μ̃ = (μ+ε)/((1-ζ)μ + ζ + ε)
Eq. (15) f = Π_d μ̃                (product T-norm; log-domain)
Eq. (16) φ = f / Σ f              (softmax(log f))
Eq. (9)  α̃ = ζ · α
Eq. (17) y_{r,c} = b_{r,c} + Σ_d α̃_{r,c,d} x_d
Eq. (19) p = softmax(Σ_r φ_r y_r - θ)

Training: Algorithm 5 (KNN init) + Algorithm 3 (GqLM): paper-aligned
GqLM under a positive-width reparameterization (raw_spreads → softplus),
with autograd probability Jacobians and float64 (default) damped
Eqs. (30)–(31). The packed parameter vector is not identical to Eq. (33).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, nn

from model.label_utils import label_key, stable_unique


ArrayLike = Union[np.ndarray, torch.Tensor, pd.DataFrame, Sequence]


def _inv_softplus(target: float) -> float:
    return float(math.log(math.expm1(max(float(target), 1e-6))))


def _cross_entropy_from_probs(probs: Tensor, targets_oh: Tensor) -> Tensor:
    """Multiclass cross-entropy from probabilities (safe log)."""
    log_p = torch.log(probs.clamp_min(torch.finfo(probs.dtype).tiny))
    return -(targets_oh * log_p).sum(dim=1).mean()


class UNFIS(nn.Module):
    """UNFIS-c: unstructured fuzzy rules with feature selection (classification).

    Parameter layout
    ----------------
    centers / raw_spreads / selector_logits : [R, D]
    consequent_weights : [R, C, D]
    consequent_bias : [R, C]
    class_thresholds : [C]
    """

    N_LINGUISTIC_CATEGORIES = 4
    PAPER_RULE_COUNT = 2

    def __init__(
        self,
        in_features: int,
        rules: int = 2,
        out_features: int = 2,
        binary: bool = False,
        drop_out_p: float = 0.0,
        selection_epsilon: float = 1e-6,
        sigma_min: float = 1e-3,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        in_features = int(in_features)
        rules = int(rules)
        out_features = int(out_features)
        selection_epsilon = float(selection_epsilon)
        sigma_min = float(sigma_min)
        drop_out_p = float(drop_out_p)

        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if rules < 1:
            raise ValueError(f"rules must be >= 1, got {rules}")
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        if selection_epsilon <= 0.0:
            raise ValueError(f"selection_epsilon must be > 0, got {selection_epsilon}")
        if sigma_min <= 0.0:
            raise ValueError(f"sigma_min must be > 0, got {sigma_min}")
        if not (0.0 <= drop_out_p <= 1.0):
            raise ValueError(f"drop_out_p must be in [0, 1], got {drop_out_p}")

        self.binary = bool(binary)
        self.in_features = in_features
        self.rules_count = rules
        self.selection_epsilon = selection_epsilon
        self.sigma_min = sigma_min
        self.drop_out_p = drop_out_p
        self.device = device

        # UNFIS-c: one output per class; binary → exactly two outputs.
        if binary:
            if out_features not in (1, 2):
                raise ValueError(
                    "Binary UNFIS-c requires out_features in {1, 2} "
                    f"(resolved to 2); got {out_features}."
                )
            self.out_features = 2
        else:
            self.out_features = out_features

        # Protocol / evaluator flags
        self.uses_custom_fit = True
        self.uses_generic_optimizer = False
        self.supports_backprop_training = False
        self.uses_reconstruction_loss = False
        self.uses_reconstruction = False
        self.uses_unfis_gqlm = True
        self.uses_unfis_protocol = True
        self.training_mode = "knn_initialization_gqlm"
        self.implementation_status = (
            "paper_aligned_independent_reimplementation_"
            "gqlm_positive_width_reparameterization"
        )

        # Antecedents [R, D]
        self.centers = nn.Parameter(torch.rand(rules, in_features, **factory_kwargs))
        raw0 = _inv_softplus(1.0)
        self.raw_spreads = nn.Parameter(
            torch.full((rules, in_features), raw0, **factory_kwargs)
        )
        self.selector_logits = nn.Parameter(
            torch.zeros(rules, in_features, **factory_kwargs)
        )

        # Consequents [R, C, D] / [R, C] / [C]
        self.consequent_weights = nn.Parameter(
            torch.zeros(rules, self.out_features, in_features, **factory_kwargs)
        )
        self.consequent_bias = nn.Parameter(
            torch.zeros(rules, self.out_features, **factory_kwargs)
        )
        self.class_thresholds = nn.Parameter(
            torch.zeros(self.out_features, **factory_kwargs)
        )

        # Kept for legacy state_dict compatibility only — never used in forward.
        self.drop_out = nn.Dropout(p=drop_out_p)
        self.decoder_linear = nn.Linear(rules, in_features, bias=True, **factory_kwargs)
        for p in self.decoder_linear.parameters():
            p.requires_grad_(False)

        self._classes: Optional[np.ndarray] = None
        self.is_fitted_ = False
        self.training_metadata: Dict[str, Any] = {}
        self.init_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Compatibility aliases (read-only)
    # ------------------------------------------------------------------

    @property
    def mean(self) -> Tensor:
        return self.centers

    @property
    def raw_std(self) -> Tensor:
        return self.raw_spreads

    @property
    def s(self) -> Tensor:
        return self.selector_logits

    @property
    def alpha(self) -> Tensor:
        return self.consequent_weights

    @property
    def theta(self) -> Tensor:
        return self.class_thresholds

    @property
    def spreads(self) -> Tensor:
        """σ = softplus(raw).clamp_min(sigma_min) (strictly positive)."""
        return F.softplus(self.raw_spreads).clamp_min(self.sigma_min)

    @property
    def std(self) -> Tensor:
        return self.spreads

    # ------------------------------------------------------------------
    # Eqs. (6), (14), (8)
    # ------------------------------------------------------------------

    def selection_strengths(self) -> Tensor:
        """Eq. (6): ζ = sigmoid(s), shape [R, D]. ζ→1 selected, ζ→0 don't-care."""
        return torch.sigmoid(self.selector_logits)

    def membership(self, x: Tensor) -> Tensor:
        """Eq. (14) Gaussian MF, shape [B, R, D]."""
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"x must be [B, {self.in_features}], got {tuple(x.shape)}"
            )
        sigma = self.spreads.clamp_min(self.sigma_min)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        return torch.exp(-(diff ** 2) / (2.0 * sigma.unsqueeze(0) ** 2))

    def selected_memberships(self, x: Tensor) -> Tensor:
        """Eq. (8) fuzzy-selection neuron, shape [B, R, D]."""
        mu = self.membership(x)
        zeta = self.selection_strengths().unsqueeze(0)
        eps = self.selection_epsilon
        return (mu + eps) / ((1.0 - zeta) * mu + zeta + eps)

    # ------------------------------------------------------------------
    # Eqs. (15)–(16)
    # ------------------------------------------------------------------

    def log_firing_strengths(self, x: Tensor) -> Tensor:
        """Stable log of product T-norm firing (Eq. 15), shape [B, R]."""
        mu_sel = self.selected_memberships(x)
        tiny = torch.finfo(mu_sel.dtype).tiny
        return torch.log(mu_sel.clamp_min(tiny)).sum(dim=2)

    def firing_strengths(self, x: Tensor) -> Tensor:
        """Raw product firing (may underflow in very high-D). Prefer log/softmax."""
        return torch.exp(self.log_firing_strengths(x))

    def normalized_firing_strengths(self, x: Tensor) -> Tensor:
        """Eq. (16): φ = softmax(log f) across rules, shape [B, R]."""
        return F.softmax(self.log_firing_strengths(x), dim=1)

    def encode(self, x: Tensor) -> Tensor:
        return self.normalized_firing_strengths(x)

    # ------------------------------------------------------------------
    # Eqs. (9), (17), (19)
    # ------------------------------------------------------------------

    def local_rule_outputs(self, x: Tensor) -> Tensor:
        """Eq. (17) with Eq. (9) gating: [B, R, C]. Bias not gated."""
        zeta = self.selection_strengths()  # [R, D]
        # α̃_{r,c,d} = ζ_{r,d} * α_{r,c,d}
        selected_w = self.consequent_weights * zeta.unsqueeze(1)
        linear = torch.einsum("bd,rcd->brc", x, selected_w)
        return self.consequent_bias.unsqueeze(0) + linear

    def class_scores(self, x: Tensor) -> Tensor:
        """Σ_r φ_r y_r − θ, shape [B, C] (pre-softmax)."""
        phi = self.normalized_firing_strengths(x)
        local = self.local_rule_outputs(x)
        scores = torch.einsum("br,brc->bc", phi, local)
        return scores - self.class_thresholds.view(1, -1)

    def probabilities(self, x: Tensor) -> Tensor:
        """Soft-Max class probabilities from class scores.

        Uses standard exponential softmax. The paper labels this Soft-Max and
        its Jacobian formulas use the usual softmax derivative; the printed
        Eq. (19) looks like a bare ratio without visible exp terms, which we
        treat as a typesetting ambiguity rather than raw score division.
        """
        return F.softmax(self.class_scores(x), dim=1)

    def forward(self, x: Tensor):
        """Return (class_scores, x.detach()) for Evaluator compatibility."""
        return self.class_scores(x), x.detach()

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def linguistic_richness(self, per_rule: bool = False):
        """Structural zero — no relational equal/not-equal/greater/less gate."""
        with torch.no_grad():
            ent = torch.zeros(
                self.rules_count, dtype=torch.float64, device=self.centers.device
            )
        if per_rule:
            return 0.0, ent
        return 0.0

    def relaxation_rate(self, per_rule: bool = False):
        """Mean of (1 − ζ). Higher ⇒ more features treated as don't-care."""
        with torch.no_grad():
            zeta = self.selection_strengths()
            per = (1.0 - zeta).mean(dim=1)
            mean = float(per.mean().item())
        if per_rule:
            return mean, per
        return mean

    def selection_matrix(self) -> Tensor:
        with torch.no_grad():
            return self.selection_strengths().detach()

    def active_feature_counts(self, threshold: Optional[float] = None) -> Tensor:
        """Per-rule active-feature measure, shape [R].

        Default (paper-style): ``zeta.sum(dim=features)`` soft count in [0, D].
        If ``threshold`` is set, hard-count features with ζ ≥ threshold (reporting).
        """
        with torch.no_grad():
            zeta = self.selection_strengths()
            if threshold is None:
                return zeta.sum(dim=1).to(dtype=torch.float64)
            return (zeta >= float(threshold)).sum(dim=1).to(dtype=torch.float64)

    # ------------------------------------------------------------------
    # Parameter packing (GqLM)
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [
            self.selector_logits,
            self.centers,
            self.raw_spreads,
            self.consequent_weights,
            self.consequent_bias,
            self.class_thresholds,
        ]

    def pack_parameters(self) -> Tensor:
        return torch.cat([p.reshape(-1) for p in self.trainable_parameters()])

    def unpack_parameters(self, vec: Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for p in self.trainable_parameters():
                n = p.numel()
                p.copy_(vec[offset : offset + n].view_as(p))
                offset += n

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "classes": None
            if self._classes is None
            else [label_key(c) if not isinstance(c, str) else c for c in np.asarray(self._classes).tolist()],
            "is_fitted_": bool(self.is_fitted_),
            "training_metadata": dict(self.training_metadata),
            "init_metadata": dict(self.init_metadata),
            "selection_epsilon": float(self.selection_epsilon),
            "sigma_min": float(self.sigma_min),
            "binary": bool(self.binary),
            "out_features": int(self.out_features),
            "rules_count": int(self.rules_count),
            "in_features": int(self.in_features),
            "implementation_status": self.implementation_status,
            "training_mode": self.training_mode,
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        classes = state.get("classes")
        self._classes = None if classes is None else np.asarray(classes)
        self.is_fitted_ = bool(state.get("is_fitted_", False))
        self.training_metadata = dict(state.get("training_metadata") or {})
        self.init_metadata = dict(state.get("init_metadata") or {})
        if "selection_epsilon" in state:
            self.selection_epsilon = float(state["selection_epsilon"])
        if "sigma_min" in state:
            self.sigma_min = float(state["sigma_min"])
        if "binary" in state:
            self.binary = bool(state["binary"])
        if "implementation_status" in state:
            self.implementation_status = state["implementation_status"]
        if "training_mode" in state:
            self.training_mode = state["training_mode"]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = dict(state_dict)

        # Legacy [D, R] mean / raw_std / s → [R, D]
        if "mean" in state_dict:
            if "centers" not in state_dict:
                mean = state_dict["mean"]
                if mean.ndim == 2 and mean.shape == (self.in_features, self.rules_count):
                    mean = mean.T.contiguous()
                state_dict["centers"] = mean
            state_dict.pop("mean", None)

        if "raw_std" in state_dict:
            if "raw_spreads" not in state_dict:
                raw = state_dict["raw_std"]
                if raw.ndim == 2 and raw.shape == (self.in_features, self.rules_count):
                    raw = raw.T.contiguous()
                state_dict["raw_spreads"] = raw
            state_dict.pop("raw_std", None)

        if "std" in state_dict:
            if "raw_spreads" not in state_dict:
                old = state_dict["std"]
                if old.ndim == 2 and old.shape == (self.in_features, self.rules_count):
                    old = old.T.contiguous()
                sigma = old.abs().clamp_min(self.sigma_min)
                state_dict["raw_spreads"] = torch.log(torch.expm1(sigma))
                warnings.warn(
                    "Converted legacy UNFIS 'std' to softplus 'raw_spreads'.",
                    RuntimeWarning,
                )
            state_dict.pop("std", None)

        if "s" in state_dict:
            if "selector_logits" not in state_dict:
                s = state_dict["s"]
                # (1, D, R) or (D, R) → (R, D)
                if s.ndim == 3 and s.shape[0] == 1:
                    s = s.squeeze(0)
                if s.ndim == 2 and s.shape == (self.in_features, self.rules_count):
                    s = s.T.contiguous()
                state_dict["selector_logits"] = s
            state_dict.pop("s", None)

        if "consequent_weight" in state_dict:
            if "consequent_weights" not in state_dict:
                state_dict["consequent_weights"] = state_dict["consequent_weight"]
            state_dict.pop("consequent_weight", None)

        if "theta" in state_dict:
            if "class_thresholds" not in state_dict:
                state_dict["class_thresholds"] = state_dict["theta"]
            state_dict.pop("theta", None)

        for key in list(state_dict.keys()):
            if key.startswith("tsk_linear."):
                state_dict.pop(key)

        if "_extra_state" not in state_dict:
            state_dict["_extra_state"] = self.get_extra_state()

        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)


# ======================================================================
# Algorithm 5 — KNN initialization
# ======================================================================


def knn_initialize_unfis(
    model: UNFIS,
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 0,
    selector_init_std: float = 0.01,
    n_neighbors: Optional[int] = None,
) -> UNFIS:
    """Algorithm 5: KNN density representatives on Z=[X, Y*].

    Documented deterministic choices where the paper is underspecified:
    * Cluster for rule i = representative V_i plus its K nearest neighbors
      in Z-space (before removal).
    * Centers use the feature coordinates of V_i (paper: m_ij ← V_i,j).
    * Class thresholds θ start at 0 (not listed in Alg. 5 outputs).
    * Remaining consequent slopes ~ N(0, 0.01).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.shape}")
    n, d = X.shape
    if n < 1:
        raise ValueError("KNN init requires at least one training sample.")
    if d != model.in_features:
        raise ValueError(
            f"X has {d} features, model expects {model.in_features}"
        )

    R = model.rules_count
    C = model.out_features
    rng = np.random.RandomState(int(random_state))

    classes = stable_unique(y)
    if model.binary and len(classes) != 2:
        raise ValueError("Binary UNFIS-c requires exactly two classes.")
    if len(classes) != C:
        raise ValueError(
            f"out_features={C} but training has {len(classes)} classes."
        )
    model._classes = classes
    class_to_idx = {label_key(c): i for i, c in enumerate(classes.tolist())}
    Y = np.zeros((n, C), dtype=np.float64)
    for i, yi in enumerate(y):
        Y[i, class_to_idx[label_key(yi)]] = 1.0

    Z = np.hstack([X, Y])
    # Paper: K = N / R
    K = max(1, int(round(n / max(R, 1))))
    K = min(K, max(n - 1, 1)) if n > 1 else 1
    if n_neighbors is not None:
        K = max(1, min(int(n_neighbors), max(n - 1, 1)))

    remaining = list(range(n))
    centers = np.zeros((R, d), dtype=np.float64)
    sigmas = np.ones((R, d), dtype=np.float64) * model.sigma_min
    biases = np.zeros((R, C), dtype=np.float64)
    cluster_sizes = []

    for r in range(R):
        if not remaining:
            # Exhausted points: duplicate last center with tiny noise.
            centers[r] = centers[max(r - 1, 0)] + rng.normal(0, 1e-6, size=d)
            sigmas[r] = model.sigma_min
            biases[r] = 1.0 / C
            cluster_sizes.append(0)
            continue

        Z_rem = Z[remaining]
        if len(remaining) == 1:
            dens_err = np.array([0.0])
            knn_idx_local = np.zeros((1, 1), dtype=np.int64)
        else:
            k_use = min(K, len(remaining) - 1)
            nn = NearestNeighbors(n_neighbors=k_use + 1, algorithm="auto").fit(Z_rem)
            dists, inds = nn.kneighbors(Z_rem)
            # Exclude self (column 0)
            dens_err = (dists[:, 1:] ** 2).mean(axis=1)
            knn_idx_local = inds[:, 1:]

        # V_i ← argmin_k e_k among remaining
        local_best = int(np.argmin(dens_err))
        global_best = remaining[local_best]
        centers[r] = X[global_best]

        # Cluster = V_i ∪ KNN(V_i)
        if len(remaining) == 1:
            members_local = [local_best]
        else:
            members_local = [local_best] + knn_idx_local[local_best].tolist()
        members_global = [remaining[j] for j in members_local]
        members_global = list(dict.fromkeys(members_global))  # unique, keep order
        cluster_X = X[members_global]
        cluster_Y = Y[members_global]
        cluster_sizes.append(len(members_global))

        if cluster_X.shape[0] == 1:
            sigmas[r] = model.sigma_min
        else:
            sigmas[r] = np.maximum(cluster_X.std(axis=0, ddof=0), model.sigma_min)

        # α0 from average desired outputs in the cluster
        biases[r] = cluster_Y.mean(axis=0)

        # Remove V_i and its KNN from Z
        remove_set = set(members_global)
        remaining = [i for i in remaining if i not in remove_set]

    # Remaining consequent slopes ~ N(0, 0.01); selector ~ N(0, selector_init_std)
    slopes = rng.normal(0.0, 0.01, size=(R, C, d))
    sel = rng.normal(0.0, float(selector_init_std), size=(R, d))

    kw = {"device": model.centers.device, "dtype": model.centers.dtype}
    with torch.no_grad():
        model.centers.copy_(torch.as_tensor(centers, **kw))
        raw = np.log(np.expm1(np.maximum(sigmas, 1e-6)))
        model.raw_spreads.copy_(torch.as_tensor(raw, **kw))
        model.consequent_bias.copy_(torch.as_tensor(biases, **kw))
        model.consequent_weights.copy_(torch.as_tensor(slopes, **kw))
        model.selector_logits.copy_(torch.as_tensor(sel, **kw))
        model.class_thresholds.zero_()

    model.init_metadata = {
        "method": "algorithm_5_knn_density",
        "K": int(K),
        "n_samples": int(n),
        "rules": int(R),
        "selector_init_std": float(selector_init_std),
        "cluster_sizes": cluster_sizes,
        "thresholds_init": "zeros",
    }
    return model


# ======================================================================
# Algorithm 3 — GqLM
# ======================================================================


def _build_probability_jacobian(
    model: UNFIS,
    X: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Autograd Jacobian of class probabilities w.r.t. packed params: [B*C, P].

    Matches paper Eq. (22) (∂p̂/∂π). Analytical Eqs. (33)–(38) are not
    hand-coded; autograd provides exact derivatives of this architecture.
    """
    params = model.trainable_parameters()
    probs = model.probabilities(X)
    flat = probs.reshape(-1)
    rows = []
    for i in range(flat.numel()):
        grads = torch.autograd.grad(
            flat[i],
            params,
            retain_graph=True,
            allow_unused=True,
        )
        parts = []
        for p, g in zip(params, grads):
            if g is None:
                parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            else:
                parts.append(g.reshape(-1))
        rows.append(torch.cat(parts))
    return torch.stack(rows, dim=0), probs.detach()


def _resolve_solver_dtype(solver_dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(solver_dtype, torch.dtype):
        if solver_dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"Unsupported solver_dtype {solver_dtype}; use float32 or float64."
            )
        return solver_dtype
    key = str(solver_dtype).lower().replace("torch.", "")
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported solver_dtype={solver_dtype!r}; expected 'float32' or 'float64'."
        )
    return mapping[key]


def _solve_damped_normal(
    J: Tensor,
    residual: Optional[Tensor] = None,
    lambda_: float = 1e3,
    *,
    rhs: Optional[Tensor] = None,
    n_classes: int = 1,
    solver_dtype: Union[str, torch.dtype] = "float64",
) -> Tuple[Tensor, str]:
    """Solve H δ = rhs with class-wise damping for stacked Jacobians.

    For stacked J (all classes), Eq. (31) is equivalent to::

        H = J.T @ J + C * lambda * I

    because Σ_c (J_c.T @ J_c + λ I) = (Σ_c J_c.T @ J_c) + C λ I.
    RHS (Eq. 30, unscaled by 1/η) is Σ_c J_c.T Ξ_c P_c^* via ``rhs``.
    If only ``residual`` is given, uses δ = -H^{-1} J^T r (MSE-style).
    """
    if int(n_classes) < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    if float(lambda_) < 0.0:
        raise ValueError(f"lambda_ must be >= 0, got {lambda_}")

    dtype = _resolve_solver_dtype(solver_dtype)
    Js = J.detach().to(dtype=dtype)
    JT = Js.transpose(0, 1)
    H = JT @ Js
    damping = float(n_classes) * float(lambda_)
    H = H + damping * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    if rhs is not None:
        rhs_s = rhs.detach().to(dtype=dtype).reshape(-1)
    elif residual is not None:
        r_s = residual.detach().to(dtype=dtype).reshape(-1)
        rhs_s = -(JT @ r_s)
    else:
        raise ValueError("Provide either rhs (Eq. 30) or residual.")

    # 1) Cholesky
    try:
        L = torch.linalg.cholesky(H)
        delta = torch.cholesky_solve(rhs_s.unsqueeze(1), L).squeeze(1)
        if torch.isfinite(delta).all():
            return delta, "cholesky"
    except Exception:
        pass

    # 2) Dense solve
    try:
        delta = torch.linalg.solve(H, rhs_s)
        if torch.isfinite(delta).all():
            return delta, "solve"
    except Exception:
        pass

    # 3) Least squares / pinv fallback
    delta = torch.linalg.lstsq(H, rhs_s.unsqueeze(1)).solution.squeeze(1)
    if not torch.isfinite(delta).all():
        delta = torch.linalg.pinv(H) @ rhs_s
    if not torch.isfinite(delta).all():
        raise RuntimeError("GqLM linear solve produced non-finite delta.")
    return delta, "lstsq"


def gqlm_train_unfis(
    model: UNFIS,
    X: ArrayLike,
    y: ArrayLike,
    *,
    minibatch_size: int = 32,
    lambda_: float = 1e3,
    eta: float = 1e-3,
    beta: float = 0.9,
    max_iterations: int = 100,
    random_state: int = 0,
    jacobian_mode: str = "autograd",
    solver_dtype: str = "float64",
    device=None,
    diagnostic_batch_size: Optional[int] = None,
) -> UNFIS:
    """Algorithm 3 GqLM trainer.

    Hyperparameter defaults ``λ=1e3``, ``η=1e−3``, ``β=0.9``, ``iter_max=100``,
    ``M=32`` match paper Table 2 (benchmark settings). ``η`` enters Eq. (30)
    as a **denominator** (Δπ scaled by ``1/η``), not a multiplier.

    Update structure (paper Alg. 3):
        Δπ* ← β Δπ* + (1 − β) Δπ
        π ← π + Δπ*

    where Δπ comes from Eqs. (30)–(31) on the mini-batch, scaled by
    ``1 / η`` according to Eq. (30). Jacobians use autograd of class
    probabilities (Eq. (22)); Ξ uses ``1/p̂`` (Eq. (26)). Status:
    paper-aligned GqLM under a positive-width reparameterization
    (autograd J; not hand-coded Eqs. (33)–(38)). Mini-batch CE is not
    globally monotone.
    """
    solve_dtype = _resolve_solver_dtype(solver_dtype)

    if int(minibatch_size) < 1:
        raise ValueError(f"minibatch_size must be >= 1, got {minibatch_size}")
    if float(lambda_) < 0.0:
        raise ValueError(f"lambda_ must be >= 0, got {lambda_}")
    if float(eta) <= 0.0:
        raise ValueError("eta must be > 0")
    if not (0.0 <= float(beta) < 1.0):
        raise ValueError(f"beta must satisfy 0 <= beta < 1, got {beta}")
    if int(max_iterations) < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

    if jacobian_mode not in ("autograd", "analytic"):
        raise ValueError(f"Unknown jacobian_mode: {jacobian_mode}")
    if jacobian_mode == "analytic":
        raise NotImplementedError(
            "Analytic Jacobian Eqs. (33)–(38) are not hand-coded; "
            "use jacobian_mode='autograd'."
        )

    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy().astype(np.float64)
    elif isinstance(X, pd.DataFrame):
        X_np = X.to_numpy(dtype=np.float64)
    else:
        X_np = np.asarray(X, dtype=np.float64)
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy().reshape(-1)
    elif isinstance(y, pd.Series):
        y_np = y.to_numpy().reshape(-1)
    else:
        y_np = np.asarray(y).reshape(-1)

    if X_np.ndim != 2 or X_np.shape[0] < 1:
        raise ValueError("GqLM requires non-empty 2-D X.")
    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("X/y length mismatch.")
    if not np.isfinite(X_np).all():
        raise ValueError("X contains non-finite values.")

    if device is not None:
        model = model.to(device)
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    model.train()

    if model._classes is None:
        knn_initialize_unfis(model, X_np, y_np, random_state=random_state)

    classes = np.asarray(model._classes)
    class_to_idx = {label_key(c): i for i, c in enumerate(classes.tolist())}
    encoded = np.asarray([class_to_idx[label_key(v)] for v in y_np], dtype=np.int64)
    Y_oh = np.eye(model.out_features, dtype=np.float64)[encoded]

    X_t = torch.as_tensor(X_np, dtype=model_dtype)  # CPU
    Y_t = torch.as_tensor(Y_oh, dtype=model_dtype)

    n = X_t.shape[0]
    bs = max(1, min(int(minibatch_size), n))
    diag_bs = (
        int(diagnostic_batch_size)
        if diagnostic_batch_size is not None
        else max(bs, 256)
    )
    diag_bs = max(1, min(diag_bs, n))
    rng = np.random.RandomState(int(random_state))
    n_classes = int(model.out_features)

    delta_star = torch.zeros(
        model.pack_parameters().numel(),
        dtype=solve_dtype,
        device=model_device,
    )
    loss_history: List[float] = []
    update_norm_history: List[float] = []
    accepted_update_history: List[bool] = []
    linear_solver_fallback_count = 0
    nonfinite_rejection_count = 0
    iterations_completed = 0

    pin = model_device.type == "cuda"

    def _full_ce() -> float:
        """Exact mean CE over the full train set (batched; no grad)."""
        total = 0.0
        count = 0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(0, n, diag_bs):
                    xb = X_t[start : start + diag_bs].to(
                        model_device, non_blocking=pin
                    )
                    yb = Y_t[start : start + diag_bs].to(
                        model_device, non_blocking=pin
                    )
                    probs = model.probabilities(xb)
                    log_p = torch.log(
                        probs.clamp_min(torch.finfo(probs.dtype).tiny)
                    )
                    # Sum over samples (and classes), then weight by batch size.
                    batch_sum = float((-(yb * log_p).sum()).item())
                    total += batch_sum
                    count += int(xb.shape[0])
        finally:
            if was_training:
                model.train()
        if count == 0:
            return float("nan")
        return total / float(count)

    ce0 = _full_ce()

    for it in range(int(max_iterations)):
        order = rng.permutation(n)
        index = 0
        while index < n:
            batch_idx = order[index : index + bs]
            index += bs
            Xb = X_t[batch_idx].to(model_device, dtype=model_dtype, non_blocking=pin)
            Yb = Y_t[batch_idx].to(model_device, dtype=model_dtype, non_blocking=pin)

            model.zero_grad(set_to_none=True)
            # Eq. (22): J stacks ∂p̂ / ∂π ; Eq. (26): Ξ_c = diag(1/p̂_c)
            J, probs = _build_probability_jacobian(model, Xb)
            with torch.no_grad():
                if not torch.isfinite(probs).all():
                    nonfinite_rejection_count += 1
                    continue
                # rhs = Σ_c J_c^T Ξ_c P_c^*  (elementwise P^*/p̂, matching J flatten)
                xi_pstar = (Yb / probs.clamp_min(1e-12)).reshape(-1)

            try:
                # Eq. (31): H = J^T J + C λ I for stacked J
                Js = J.detach().to(dtype=solve_dtype)
                rhs = Js.transpose(0, 1) @ xi_pstar.to(dtype=solve_dtype)
                delta, solver = _solve_damped_normal(
                    J,
                    residual=None,
                    lambda_=float(lambda_),
                    rhs=rhs,
                    n_classes=n_classes,
                    solver_dtype=solve_dtype,
                )
            except RuntimeError:
                nonfinite_rejection_count += 1
                continue
            if solver != "cholesky":
                linear_solver_fallback_count += 1

            # Eq. (30): Δπ = (1/η) * J+ * Σ_c J_c^T Ξ_c P_c^*
            delta = delta / float(eta)
            if not torch.isfinite(delta).all():
                nonfinite_rejection_count += 1
                continue

            # Alg. 3: Δπ* ← β Δπ* + (1 − β) Δπ
            delta = delta.to(device=delta_star.device, dtype=delta_star.dtype)
            delta_star = float(beta) * delta_star + (1.0 - float(beta)) * delta
            if not torch.isfinite(delta_star).all():
                nonfinite_rejection_count += 1
                delta_star = torch.zeros_like(delta_star)
                continue

            theta = model.pack_parameters().detach()
            candidate = theta + delta_star.to(device=theta.device, dtype=theta.dtype)
            if not torch.isfinite(candidate).all():
                nonfinite_rejection_count += 1
                continue

            model.unpack_parameters(candidate)
            # Softplus keeps spreads positive by construction.
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                model.unpack_parameters(theta)
                nonfinite_rejection_count += 1
                continue

            update_norm_history.append(float(delta_star.norm().item()))
            accepted_update_history.append(True)

        iterations_completed = it + 1
        loss_history.append(_full_ce())

    model.eval()
    model.is_fitted_ = True
    ce_final = _full_ce()
    model.training_metadata = {
        "optimizer": "GqLM",
        "scheduler": None,
        "jacobian_mode": "autograd",
        "implementation_note": (
            "Algorithm 3 with autograd probability Jacobians and "
            "positive-width reparameterization."
        ),
        "gqlm_equation_30_scale": "1/eta",
        "gqlm_equation_31_damping": "C*lambda_for_stacked_jacobian",
        "eta_interpretation": "denominator_as_in_eq_30",
        "eta_source": "paper",
        "parameterization": "raw_spreads_softplus",
        "solver_dtype": (
            "float64" if solve_dtype == torch.float64 else "float32"
        ),
        "minibatch_size": bs,
        "diagnostic_batch_size": diag_bs,
        "lambda": float(lambda_),
        "eta": float(eta),
        "beta": float(beta),
        "max_iterations": int(max_iterations),
        "n_classes_damping": n_classes,
        "ce_initial": ce0,
        "ce_final": ce_final,
        "loss_history": loss_history,
        "update_norm_history": update_norm_history,
        "accepted_update_history": accepted_update_history,
        "linear_solver_fallback_count": int(linear_solver_fallback_count),
        "nonfinite_rejection_count": int(nonfinite_rejection_count),
        "iterations_completed": int(iterations_completed),
        "hyperparameter_source": "paper_table_2",
    }
    return model


def train_unfis_model(
    model: UNFIS,
    X: ArrayLike,
    y: ArrayLike,
    *,
    random_state: int = 0,
    minibatch_size: int = 32,
    lambda_: float = 1e3,
    eta: float = 1e-3,
    beta: float = 0.9,
    max_iterations: int = 100,
    selector_init_std: float = 0.01,
    jacobian_mode: str = "autograd",
    solver_dtype: str = "float64",
    device=None,
) -> UNFIS:
    """KNN initialization (Alg. 5) followed by GqLM (Alg. 3)."""
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy().astype(np.float64)
    elif isinstance(X, pd.DataFrame):
        X_np = X.to_numpy(dtype=np.float64)
    else:
        X_np = np.asarray(X, dtype=np.float64)
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy().reshape(-1)
    elif isinstance(y, pd.Series):
        y_np = y.to_numpy().reshape(-1)
    else:
        y_np = np.asarray(y).reshape(-1)

    knn_initialize_unfis(
        model,
        X_np,
        y_np,
        random_state=random_state,
        selector_init_std=selector_init_std,
    )
    # Paper Table 2 benchmark defaults: λ=1e3, η=1e−3, β=0.9, iter=100.
    # η is the Eq. (30) denominator (Δπ ∝ 1/η), not a multiplicative LR.
    return gqlm_train_unfis(
        model,
        X_np,
        y_np,
        minibatch_size=minibatch_size,
        lambda_=lambda_,
        eta=eta,
        beta=beta,
        max_iterations=max_iterations,
        random_state=random_state,
        jacobian_mode=jacobian_mode,
        solver_dtype=solver_dtype,
        device=device,
    )


# ======================================================================
# Sklearn wrapper
# ======================================================================


class SklearnUNFISWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn facade for UNFIS-c (KNN init + GqLM)."""

    def __init__(
        self,
        model,
        device=None,
        dtype=torch.float32,
        batch_size: int = 1024,
    ):
        self.device = device if device else "cpu"
        self.dtype = dtype
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size
        self.model = model.to(device=self.device, dtype=self.dtype)
        if getattr(self.model, "is_fitted_", False) and self.model._classes is not None:
            self.classes_ = np.asarray(self.model._classes)
            self.is_fitted_ = True
        else:
            self.classes_ = getattr(self.model, "_classes", None)
            self.is_fitted_ = bool(getattr(self.model, "is_fitted_", False))

    def fit(
        self,
        X,
        y,
        *,
        random_state: int = 0,
        minibatch_size: int = 32,
        lambda_: float = 1e3,
        eta: float = 1e-3,
        beta: float = 0.9,
        max_iterations: int = 100,
        selector_init_std: float = 0.01,
        jacobian_mode: str = "autograd",
        solver_dtype: str = "float64",
        device=None,
        verbose: bool = False,
        **kwargs,
    ):
        del verbose, kwargs  # evaluator may pass unused keys
        if device is not None:
            self.device = device
            self.model = self.model.to(self.device)
        train_unfis_model(
            self.model,
            X,
            y,
            random_state=random_state,
            minibatch_size=minibatch_size,
            lambda_=lambda_,
            eta=eta,
            beta=beta,
            max_iterations=max_iterations,
            selector_init_std=selector_init_std,
            jacobian_mode=jacobian_mode,
            solver_dtype=solver_dtype,
            device=self.device,
        )
        self.classes_ = np.asarray(self.model._classes)
        self.is_fitted_ = True
        return self

    def unfis_gqlm(self, X, y, **kwargs):
        """Evaluator ``fit_method`` alias for :meth:`fit`."""
        return self.fit(X, y, **kwargs)

    def _convert_to_tensor(self, data) -> Tensor:
        if isinstance(data, np.ndarray):
            return torch.as_tensor(data, dtype=self.dtype)
        if isinstance(data, torch.Tensor):
            return data.detach().to(device="cpu", dtype=self.dtype)
        if isinstance(data, pd.DataFrame):
            return torch.as_tensor(data.to_numpy(), dtype=self.dtype)
        raise ValueError("Input must be NumPy, Tensor, or DataFrame.")

    def _iter_batches(self, X: Tensor):
        n = X.shape[0]
        bs = max(1, self.batch_size)
        for start in range(0, n, bs):
            yield X[start : start + bs]

    def _forward_scores(self, X) -> Tensor:
        X_t = self._convert_to_tensor(X)
        if X_t.shape[0] == 0:
            return torch.empty((0, self.model.out_features), dtype=self.dtype)
        model_param = next(self.model.parameters())
        model_device = model_param.device
        model_dtype = model_param.dtype
        pin = model_device.type == "cuda"
        was_training = self.model.training
        self.model.eval()
        chunks: List[Tensor] = []
        try:
            with torch.no_grad():
                for xb in self._iter_batches(X_t):
                    xb = xb.to(
                        device=model_device,
                        dtype=model_dtype,
                        non_blocking=pin,
                    )
                    chunks.append(self.model.class_scores(xb).detach().cpu())
        finally:
            if was_training:
                self.model.train()
        return torch.cat(chunks, dim=0)

    def _check_fitted(self):
        if not bool(getattr(self, "is_fitted_", False)) or getattr(
            self.model, "_classes", None
        ) is None:
            raise NotFittedError(
                "This SklearnUNFISWrapper instance is not fitted yet. "
                "Call fit() before predict()."
            )

    def decision_function(self, X):
        self._check_fitted()
        return self._forward_scores(X).numpy()

    def predict_proba(self, X):
        self._check_fitted()
        scores = self._forward_scores(X)
        if scores.shape[0] == 0:
            return np.empty((0, self.model.out_features), dtype=np.float32)
        return torch.softmax(scores, dim=1).numpy()

    def predict(self, X):
        self._check_fitted()
        scores = self._forward_scores(X)
        classes = np.asarray(self.model._classes)
        if scores.shape[0] == 0:
            return np.empty((0,), dtype=classes.dtype)
        return classes[scores.argmax(dim=1).numpy()]

    def score(self, X, y):
        self._check_fitted()
        return accuracy_score(np.asarray(y).reshape(-1), self.predict(X))

    def get_params(self, deep=True):
        return {
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
        }

    def _sync_fitted_from_model(self) -> None:
        if getattr(self.model, "is_fitted_", False) and getattr(
            self.model, "_classes", None
        ) is not None:
            self.classes_ = np.asarray(self.model._classes)
            self.is_fitted_ = True
        else:
            self.classes_ = None
            self.is_fitted_ = False

    def set_params(self, **parameters):
        for key, value in parameters.items():
            if key not in {"model", "device", "dtype", "batch_size"}:
                raise ValueError(f"Unknown parameter: {key}")
            if key == "batch_size":
                value = int(value)
                if value < 1:
                    raise ValueError(f"batch_size must be >= 1, got {value}")
            setattr(self, key, value)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self._sync_fitted_from_model()
        return self


if __name__ == "__main__":
    m = UNFIS(4, 2, 2, binary=True)
    X = torch.rand(8, 4)
    scores, xd = m(X)
    assert scores.shape == (8, 2)
    print("ok", m.training_mode, float(m.relaxation_rate()))
