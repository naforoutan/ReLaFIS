"""
An independent PyTorch reproduction of ADMTSK from the published equations.

Reference
---------
G. Xue, L. Hu, J. Wang and S. Ablameyko,
"ADMTSK: A High-Dimensional Takagi–Sugeno–Kang Fuzzy System Based on
Adaptive Dombi T-Norm," IEEE Transactions on Fuzzy Systems, vol. 33,
no. 6, pp. 1767–1780, June 2025. doi: 10.1109/TFUZZ.2025.3535640.

This is not official author code. Exact replication of published table
numbers is not claimed until validated on the paper datasets with the
full repeated-CV protocol.
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
from sklearn.metrics import accuracy_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


ArrayLike = Union[np.ndarray, torch.Tensor, pd.DataFrame, Sequence]


def _inv_softplus(target: float) -> float:
    """Inverse softplus so softplus(raw) ≈ target (target > 0)."""
    return float(math.log(math.expm1(float(target))))


def paper_mse_loss(outputs: Tensor, targets: Tensor) -> Tensor:
    """Paper eq. (9): L = (1/(2N)) Σ_n Σ_c (output - target)^2.

    Equivalent PyTorch form used throughout this repo:
        0.5 * ((outputs - targets) ** 2).sum(dim=-1).mean()

    Softmax/sigmoid are intentionally NOT applied before this loss —
    ADMTSK trains on raw TSK scores against one-hot targets.
    """
    return 0.5 * ((outputs - targets) ** 2).sum(dim=-1).mean()


def adaptive_lambda(
    n_features: int,
    K: float = 10.0,
    membership_lower_bound: float = 1.0 / math.e,
) -> float:
    """Paper adaptive Dombi index λ (eq. 30), computed once from D.

    λ = log(D) / (log(K - ε) - log(1 - ε))

    λ is not learnable: it depends only on dimensionality so the Dombi
    T-norm stays calibrated as D grows. Product T-norms underflow in
    high-D; Dombi with this λ avoids that collapse.
    """
    D = int(n_features)
    if D < 1:
        raise ValueError(f"n_features must be >= 1, got {D}")
    eps = float(membership_lower_bound)
    K = float(K)
    if not (0.0 < eps < 1.0):
        raise ValueError(f"membership_lower_bound must be in (0, 1), got {eps}")
    if K <= 1.0:
        raise ValueError(f"K must be > 1, got {K}")
    if K <= eps:
        raise ValueError(f"K must be > membership_lower_bound, got K={K}, eps={eps}")
    denom = math.log(K - eps) - math.log(1.0 - eps)
    if denom <= 0.0:
        raise ValueError(
            f"Adaptive-lambda denominator must be > 0, got {denom} "
            f"(K={K}, membership_lower_bound={eps})"
        )
    # Preserve the paper equation even for small D (λ may be < 1).
    return float(math.log(D) / denom)


class ADMTSK(nn.Module):
    """Adaptive Dombi TSK with composite Gaussian membership (CGMF).

    Compactly combined fuzzy rule base (CoCo-FRB): R rules, each rule r
    uses fuzzy set r on every feature (no R^D expansion, no clustering).

    Parameter layout
    ----------------
    centers / spreads : [R, D]
    consequent_weights : [R, C, D]
    consequent_bias : [R, C]

    ``forward`` returns ``(logits, x.detach())`` for Evaluator interface
    compatibility. ADMTSK has no reconstruction objective — the second
    value is never used as a decoder target when
    ``uses_reconstruction=False``.
    """

    N_LINGUISTIC_CATEGORIES = 4
    PAPER_RULE_COUNT = 3
    PAPER_EPOCH_COUNT = 50
    PAPER_HYPERPARAMETER_GRID = {
        "learning_rate": [0.01, 0.001, 0.0001],
        "batch_fraction": [0.1, 0.2],
    }

    def __init__(
        self,
        in_features: int,
        rules: int = 3,
        out_features: int = 1,
        binary: bool = True,
        K: float = 10.0,
        membership_lower_bound: float = 1.0 / math.e,
        min_spread: float = 1e-6,
        dombi_epsilon: float = 1e-12,
        feature_chunk_size: Optional[int] = None,
        validate_input_range: bool = False,
        dtype: torch.dtype = torch.float32,
        device=None,
        *,
        # Compatibility kwargs for experiments/protocols/admtsk.py
        drop_out_p: float = 0.0,
        adaptive: bool = True,
        lambda_: Optional[float] = None,
        paper_init: bool = True,
        paper_mode: bool = False,
        sigma_min: Optional[float] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        in_features = int(in_features)
        rules = int(rules)
        out_features = int(out_features)
        K = float(K)
        membership_lower_bound = float(membership_lower_bound)
        min_spread = float(min_spread if sigma_min is None else sigma_min)
        dombi_epsilon = float(dombi_epsilon)
        adaptive = bool(adaptive)

        if not adaptive and lambda_ is None:
            raise ValueError("Fixed-lambda mode requires a positive lambda_.")
        if adaptive and in_features < 2:
            raise ValueError(
                "ADMTSK adaptive λ requires in_features >= 2 "
                f"(got {in_features}). Pass adaptive=False and a positive "
                "lambda_ for a fixed Dombi index on lower-dimensional data."
            )
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if rules < 1:
            raise ValueError(f"rules must be >= 1, got {rules}")
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        if K <= 1.0:
            raise ValueError(f"K must be > 1, got {K}")
        if not (0.0 < membership_lower_bound < 1.0):
            raise ValueError(
                "membership_lower_bound must satisfy 0 < ε < 1, "
                f"got {membership_lower_bound}"
            )
        if min_spread <= 0.0:
            raise ValueError(f"min_spread must be > 0, got {min_spread}")
        if dombi_epsilon <= 0.0:
            raise ValueError(f"dombi_epsilon must be > 0, got {dombi_epsilon}")
        if feature_chunk_size is not None and int(feature_chunk_size) < 1:
            raise ValueError(
                f"feature_chunk_size must be >= 1 or None, got {feature_chunk_size}"
            )
        if paper_mode:
            if rules != self.PAPER_RULE_COUNT:
                raise ValueError(
                    f"paper_mode requires rules == {self.PAPER_RULE_COUNT}, got {rules}"
                )
            if not paper_init:
                raise ValueError("paper_mode requires paper_init=True")
            if not adaptive:
                raise ValueError("paper_mode requires adaptive=True")
            if abs(K - 10.0) > 1e-12:
                raise ValueError(f"paper_mode requires K == 10, got {K}")
            if not math.isclose(
                membership_lower_bound,
                1.0 / math.e,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "paper_mode requires membership_lower_bound == 1/e, "
                    f"got {membership_lower_bound}"
                )

        self.in_features = in_features
        self.rules_count = rules
        self.out_features = out_features
        self.binary = bool(binary)
        self.K = K
        self.membership_lower_bound = membership_lower_bound
        self.min_spread = min_spread
        self.dombi_epsilon = dombi_epsilon
        self.feature_chunk_size = (
            None if feature_chunk_size is None else int(feature_chunk_size)
        )
        self.validate_input_range = bool(validate_input_range)
        self.device = device
        self.drop_out_p = float(drop_out_p)
        self.adaptive = adaptive
        self.paper_init = bool(paper_init)
        self.paper_mode = bool(paper_mode)
        self.uses_reconstruction = False
        self.uses_reconstruction_loss = False

        # ---- Antecedents: centers [R, D], raw spreads [R, D] ----
        # Paper eq. (33), one-based: m_(r,d) = (r-1)/(R-1).
        # Zero-based equivalent: m[r, d] = r / (R - 1).
        if paper_init:
            if rules == 1:
                centers = torch.full((rules, in_features), 0.5, **factory_kwargs)
            else:
                grid = torch.arange(rules, **factory_kwargs) / float(rules - 1)
                centers = grid.view(rules, 1).expand(rules, in_features).contiguous()
            # σ = softplus(raw) + min_spread ≈ 1 at init
            raw_target = max(1.0 - min_spread, 1e-6)
            raw_init = _inv_softplus(raw_target)
            raw_spreads = torch.full(
                (rules, in_features), raw_init, **factory_kwargs
            )
        else:
            centers = torch.rand((rules, in_features), **factory_kwargs)
            raw_target = max(1.0 - min_spread, 1e-6)
            raw_init = _inv_softplus(raw_target)
            raw_spreads = torch.full(
                (rules, in_features), raw_init, **factory_kwargs
            ) + 0.1 * torch.randn(rules, in_features, **factory_kwargs)

        self.centers = nn.Parameter(centers)
        self.raw_spreads = nn.Parameter(raw_spreads)

        # ---- First-order TSK consequents (eq. 7): bias + W @ x ----
        self.consequent_weights = nn.Parameter(
            torch.zeros(rules, out_features, in_features, **factory_kwargs)
        )
        self.consequent_bias = nn.Parameter(
            torch.zeros(rules, out_features, **factory_kwargs)
        )

        # Adaptive λ once from D (eq. 30). Not an nn.Parameter.
        if adaptive:
            lam = adaptive_lambda(in_features, K=K, membership_lower_bound=membership_lower_bound)
            if lam < 1.0:
                warnings.warn(
                    "Adaptive ADMTSK produced lambda < 1 for "
                    f"D={in_features}: lambda={lam:.6f}. "
                    "This preserves the published equation without clamping, "
                    "but applies ADMTSK outside its primary high-dimensional regime.",
                    RuntimeWarning,
                )
        else:
            lam = float(lambda_)
            if lam <= 0.0:
                raise ValueError(f"lambda_ must be > 0, got {lam}")

        # Preserve paper λ (no clamp). Protocol still reads these fields.
        self.lambda_unclamped = float(lam)
        self.lambda_was_clamped = False
        self.register_buffer(
            "lambda_value",
            torch.tensor(float(lam), device=device, dtype=dtype),
        )

        self._classes: Optional[np.ndarray] = None
        self.is_fitted_ = False
        self.training_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Compatibility aliases (read-only; not registered parameters)
    # ------------------------------------------------------------------

    @property
    def mean(self) -> Tensor:
        return self.centers

    @property
    def raw_std(self) -> Tensor:
        return self.raw_spreads

    @property
    def consequent_weight(self) -> Tensor:
        return self.consequent_weights

    @property
    def _lambda_adaptive(self) -> Tensor:
        return self.lambda_value

    # ------------------------------------------------------------------
    # Spreads
    # ------------------------------------------------------------------

    @property
    def spreads(self) -> Tensor:
        """Effective positive spreads σ = softplus(raw) + min_spread.

        Spreads stay strictly positive without post-hoc clamping of a
        directly learnable σ after optimizer steps.
        """
        return F.softplus(self.raw_spreads) + self.min_spread

    @property
    def std(self) -> Tensor:
        return self.spreads

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------

    def _maybe_validate_input_range(self, x: Tensor) -> None:
        """Fold-local MinMax to [0, 1] is mandatory for paper CGMF centers.

        Scaler fitting belongs in the experiment pipeline (train fold only),
        never inside forward().
        """
        if not self.validate_input_range:
            return
        x_min = float(x.min().detach())
        x_max = float(x.max().detach())
        # Tolerate tiny floating-point excursions around [0, 1].
        if x_min < -1e-5 or x_max > 1.0 + 1e-5:
            raise ValueError(
                "ADMTSK expects inputs in [0, 1] (paper MinMax normalization). "
                f"Observed min={x_min}, max={x_max}. Fit MinMaxScaler on the "
                "training fold only, then transform val/test with that scaler."
            )

    # ------------------------------------------------------------------
    # CGMF membership (eq. 31)
    # ------------------------------------------------------------------

    def membership(self, x: Tensor) -> Tensor:
        """Composite Gaussian membership function (CGMF), paper eq. (31).

        μ_(r,d)(x) = exp(-1 + exp(-((x_d - m)^2) / (2 σ^2)))

        Why lower bound 1/e: as |x - m| → ∞ the inner Gaussian → 0, so
        μ → exp(-1) = 1/e. At the center, μ = 1. Range is (1/e, 1].

        This is NOT the extended Gaussian MF ε + (1-ε)·Gaussian.

        When ``feature_chunk_size`` is set, intermediates are computed in
        feature chunks, but the returned membership is still the complete
        ``[B, R, D]`` tensor. True streaming aggregation that never retains
        the full membership tensor is provided by ``firing_strengths()``.

        Parameters
        ----------
        x : [B, D]

        Returns
        -------
        mu : [B, R, D]
        """
        if x.ndim != 2:
            raise ValueError(f"x must be [B, D], got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"x has {x.shape[1]} features, model expects {self.in_features}"
            )
        self._maybe_validate_input_range(x)

        B = x.shape[0]
        R, D = self.rules_count, self.in_features
        centers = self.centers  # [R, D]
        sigma = self.spreads    # [R, D]

        chunk = self.feature_chunk_size
        if chunk is None or chunk >= D:
            # x: [B, 1, D], centers: [1, R, D]
            diff = x.unsqueeze(1) - centers.unsqueeze(0)
            g = torch.exp(-(diff ** 2) / (2.0 * sigma.unsqueeze(0) ** 2))
            return torch.exp(-1.0 + g)

        # Feature-wise chunking reduces intermediate calculation size when D
        # is huge, but the public return is still the full [B, R, D] tensor
        # (concatenated). For streaming aggregation that never retains the
        # complete membership tensor, use firing_strengths() with
        # feature_chunk_size set.
        pieces: List[Tensor] = []
        for start in range(0, D, chunk):
            end = min(start + chunk, D)
            x_c = x[:, start:end]
            c_c = centers[:, start:end]
            s_c = sigma[:, start:end]
            diff = x_c.unsqueeze(1) - c_c.unsqueeze(0)
            g = torch.exp(-(diff ** 2) / (2.0 * s_c.unsqueeze(0) ** 2))
            pieces.append(torch.exp(-1.0 + g))
        return torch.cat(pieces, dim=2)

    def cgmf(self, x: Tensor) -> Tensor:
        """Alias returning membership in legacy [B, D, R] layout."""
        return self.membership(x).transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Dombi firing (eq. 21)
    # ------------------------------------------------------------------

    def firing_strengths(self, x: Tensor) -> Tensor:
        """Adaptive Dombi firing strengths φ_r(x), paper eq. (21).

        φ_r = 1 / (1 + (Σ_d (1/μ_(r,d) - 1)^λ )^(1/λ))

        Dombi aggregation avoids product-T-norm underflow in high-D.
        Numerical guards (clamp of 1/μ-1, log-domain sum) keep gradients
        finite without changing the mathematical Dombi meaning.
        """
        self._maybe_validate_input_range(x)
        D = self.in_features
        chunk = self.feature_chunk_size
        lam = self.lambda_value.to(dtype=x.dtype, device=x.device)
        eps = self.dombi_epsilon

        if chunk is None or chunk >= D:
            mu = self.membership(x)  # [B, R, D]
            base = (1.0 / mu - 1.0).clamp_min(eps)
            # log-domain: (Σ base^λ)^(1/λ) = exp(logsumexp(λ log base) / λ)
            log_terms = lam * torch.log(base)
            log_sum = torch.logsumexp(log_terms, dim=2)
            root = torch.exp(log_sum / lam)
            return 1.0 / (1.0 + root)

        # Accumulate Σ_d base^λ across feature chunks, then take ^(1/λ).
        B = x.shape[0]
        R = self.rules_count
        centers = self.centers
        sigma = self.spreads
        powered_sum = x.new_zeros(B, R)

        for start in range(0, D, chunk):
            end = min(start + chunk, D)
            x_c = x[:, start:end]
            c_c = centers[:, start:end]
            s_c = sigma[:, start:end]
            diff = x_c.unsqueeze(1) - c_c.unsqueeze(0)
            g = torch.exp(-(diff ** 2) / (2.0 * s_c.unsqueeze(0) ** 2))
            mu_c = torch.exp(-1.0 + g)
            base = (1.0 / mu_c - 1.0).clamp_min(eps)
            powered_sum = powered_sum + base.pow(lam).sum(dim=2)

        root = powered_sum.clamp_min(eps).pow(1.0 / lam)
        return 1.0 / (1.0 + root)

    def encode(self, x: Tensor) -> Tensor:
        """Unnormalized Dombi firing strengths [B, R]."""
        return self.firing_strengths(x)

    def normalized_firing_strengths(self, x: Tensor) -> Tensor:
        """Rule-normalized firing strengths (sum to 1 over rules)."""
        phi = self.firing_strengths(x)
        denom = phi.sum(dim=1, keepdim=True).clamp_min(torch.finfo(phi.dtype).tiny)
        return phi / denom

    def dombi_firing(self, mu: Tensor, *, stable: bool = True) -> Tensor:
        """Dombi firing from a precomputed membership tensor.

        Accepts mu as [B, R, D] (preferred) or legacy [B, D, R].
        """
        if mu.ndim != 3:
            raise ValueError(f"mu must be 3-D, got {tuple(mu.shape)}")
        if mu.shape[1] == self.in_features and mu.shape[2] == self.rules_count:
            mu = mu.transpose(1, 2).contiguous()  # -> [B, R, D]
        if mu.shape[1] != self.rules_count or mu.shape[2] != self.in_features:
            raise ValueError(
                f"mu shape {tuple(mu.shape)} incompatible with "
                f"rules={self.rules_count}, features={self.in_features}"
            )
        lam = self.lambda_value.to(dtype=mu.dtype, device=mu.device)
        eps = self.dombi_epsilon
        base = (1.0 / mu - 1.0).clamp_min(eps)
        if not stable:
            s = base.pow(lam).sum(dim=2)
            return 1.0 / (1.0 + s.pow(1.0 / lam))
        log_terms = lam * torch.log(base)
        log_sum = torch.logsumexp(log_terms, dim=2)
        root = torch.exp(log_sum / lam)
        return 1.0 / (1.0 + root)

    # ------------------------------------------------------------------
    # TSK consequents (eqs. 7–8)
    # ------------------------------------------------------------------

    def rule_outputs(self, x: Tensor) -> Tensor:
        """First-order TSK rule outputs [B, R, C]. No sigmoid/softmax."""
        D = self.in_features
        chunk = self.feature_chunk_size
        bias = self.consequent_bias.unsqueeze(0)  # [1, R, C]

        if chunk is None or chunk >= D:
            # consequent_weights [R, C, D]: y_{r,c} = b_{r,c} + Σ_d W_{r,c,d} x_d
            linear = torch.einsum("bd,rcd->brc", x, self.consequent_weights)
            return bias + linear

        B = x.shape[0]
        R, C = self.rules_count, self.out_features
        acc = x.new_zeros(B, R, C)
        for start in range(0, D, chunk):
            end = min(start + chunk, D)
            acc = acc + torch.einsum(
                "bd,rcd->brc",
                x[:, start:end],
                self.consequent_weights[:, :, start:end],
            )
        return bias + acc

    def consequent(self, x: Tensor, firing: Tensor) -> Tensor:
        """Weighted sum of rule outputs with normalized firing [B, C]."""
        local = self.rule_outputs(x)
        return torch.einsum("br,brc->bc", firing, local)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor):
        """Return raw TSK logits and a detached input for Evaluator compat.

        Returns
        -------
        logits : [B, C]
        x_detached : same shape as x (not a reconstruction; no recon loss)
        """
        if x.ndim != 2:
            raise ValueError(f"x must be [B, D], got shape {tuple(x.shape)}")
        self._maybe_validate_input_range(x)
        norm_phi = self.normalized_firing_strengths(x)
        logits = self.consequent(x, norm_phi)
        return logits, x.detach()

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def linguistic_richness(self, per_rule: bool = False):
        """Structural zero — ADMTSK has no relational gate vocabulary.

        Classical CoCo-FRB CGMF terms only express “approximately equal to
        center”; there is no equal/not-equal or greater/less gate, so the
        shared 4-category Shannon entropy is exactly 0 by construction.
        """
        with torch.no_grad():
            entropies = torch.zeros(
                self.rules_count, dtype=torch.float64, device=self.centers.device
            )
        if per_rule:
            return 0.0, entropies
        return 0.0

    # relaxation_rate is intentionally absent: ADMTSK has no learned
    # “don't care” relaxation gate. The Evaluator records NaN (= N/A).
    # Returning 0.0 would falsely imply the model supports relaxation but
    # learned none.

    # ------------------------------------------------------------------
    # Checkpoint helpers (protocol compatibility)
    # ------------------------------------------------------------------

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "classes": None
            if self._classes is None
            else np.asarray(self._classes).tolist(),
            "is_fitted_": bool(self.is_fitted_),
            "training_metadata": dict(self.training_metadata),
            "binary": bool(self.binary),
            "in_features": int(self.in_features),
            "rules_count": int(self.rules_count),
            "out_features": int(self.out_features),
            "K": float(self.K),
            "lambda_unclamped": float(self.lambda_unclamped),
            "lambda_value": float(self.lambda_value.detach().cpu()),
            "lambda_was_clamped": bool(self.lambda_was_clamped),
            "min_spread": float(self.min_spread),
            "paper_mode": bool(self.paper_mode),
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        classes = state.get("classes")
        self._classes = None if classes is None else np.asarray(classes)
        self.is_fitted_ = bool(state.get("is_fitted_", False))
        self.training_metadata = dict(state.get("training_metadata") or {})
        self.lambda_unclamped = float(
            state.get("lambda_unclamped", self.lambda_unclamped)
        )
        self.lambda_was_clamped = bool(
            state.get("lambda_was_clamped", self.lambda_was_clamped)
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load weights, converting legacy alias keys then removing them."""
        state_dict = dict(state_dict)

        if "mean" in state_dict:
            if "centers" not in state_dict:
                mean = state_dict["mean"]
                if mean.shape == (self.in_features, self.rules_count):
                    mean = mean.T.contiguous()
                state_dict["centers"] = mean
            state_dict.pop("mean", None)

        if "raw_std" in state_dict:
            if "raw_spreads" not in state_dict:
                raw = state_dict["raw_std"]
                if raw.shape == (self.in_features, self.rules_count):
                    raw = raw.T.contiguous()
                state_dict["raw_spreads"] = raw
            state_dict.pop("raw_std", None)

        if "std" in state_dict:
            if "raw_spreads" not in state_dict:
                old_std = state_dict["std"]
                if old_std.shape == (self.in_features, self.rules_count):
                    old_std = old_std.T.contiguous()
                old_sigma = old_std.abs().clamp_min(self.min_spread)
                target = (old_sigma - self.min_spread).clamp_min(1e-6)
                state_dict["raw_spreads"] = torch.log(torch.expm1(target))
                warnings.warn(
                    "Converted legacy ADMTSK 'std' to softplus 'raw_spreads'.",
                    RuntimeWarning,
                )
            state_dict.pop("std", None)

        if "consequent_weight" in state_dict:
            if "consequent_weights" not in state_dict:
                state_dict["consequent_weights"] = state_dict["consequent_weight"]
            state_dict.pop("consequent_weight", None)

        if "_lambda_adaptive" in state_dict:
            if "lambda_value" not in state_dict:
                state_dict["lambda_value"] = state_dict["_lambda_adaptive"]
            state_dict.pop("_lambda_adaptive", None)

        # Older checkpoints may predate get_extra_state(); synthesize it so
        # strict loading does not fail solely on that key.
        if "_extra_state" not in state_dict:
            state_dict["_extra_state"] = self.get_extra_state()

        # PyTorch versions before ~2.1 may not accept ``assign``.
        try:
            return super().load_state_dict(
                state_dict, strict=strict, assign=assign
            )
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)


def _to_numpy_xy(X: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(X, pd.DataFrame):
        X_np = X.to_numpy(dtype=np.float64)
    elif isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy().astype(np.float64)
    else:
        X_np = np.asarray(X, dtype=np.float64)
    if X_np.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X_np.shape}")

    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    elif isinstance(y, pd.Series):
        y_np = y.to_numpy()
    else:
        y_np = np.asarray(y)
    y_np = y_np.reshape(-1)
    if y_np.shape[0] != X_np.shape[0]:
        raise ValueError(f"X/y length mismatch: {X_np.shape[0]} vs {y_np.shape[0]}")
    return X_np, y_np


def train_admtsk_model(
    model: ADMTSK,
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    learning_rate: float = 0.001,
    batch_fraction: float = 0.2,
    epochs: int = 50,
    random_state: int = 0,
    device=None,
) -> ADMTSK:
    """Paper trainer: fixed-LR Adam + one-hot MSE (eq. 9), no OneCycleLR.

    Batch size = max(1, round(batch_fraction * N_train)).
    """
    learning_rate = float(learning_rate)
    batch_fraction = float(batch_fraction)
    epochs = int(epochs)
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if not (0.0 < batch_fraction <= 1.0):
        raise ValueError(f"batch_fraction must be in (0, 1], got {batch_fraction}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    X_np, y_np = _to_numpy_xy(X_train, y_train)
    if X_np.shape[1] != model.in_features:
        raise ValueError(
            f"X has {X_np.shape[1]} features, model expects {model.in_features}"
        )

    classes = np.unique(y_np)
    if model.out_features == 1:
        # Single-logit compatibility: encode positive class as 1.0 target.
        if len(classes) != 2:
            raise ValueError("out_features=1 requires binary labels.")
        encoded = (y_np == classes[1]).astype(np.float64)
        y_target = torch.as_tensor(encoded, dtype=torch.float32).unsqueeze(1)
    else:
        if len(classes) != model.out_features:
            raise ValueError(
                f"out_features={model.out_features} but training has "
                f"{len(classes)} classes."
            )
        class_to_index = {label: i for i, label in enumerate(classes.tolist())}
        encoded = np.asarray([class_to_index[v] for v in y_np], dtype=np.int64)
        y_target = F.one_hot(
            torch.as_tensor(encoded), num_classes=model.out_features
        ).to(dtype=torch.float32)

    model._classes = np.asarray(classes)
    if device is not None:
        model = model.to(device)
    model.train()

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Keep the full dataset on CPU; move only minibatches to the model device.
    X_t = torch.as_tensor(X_np, dtype=model_dtype)
    y_t = y_target.to(dtype=model_dtype, device="cpu")

    n = X_np.shape[0]
    batch_size = max(1, int(round(batch_fraction * n)))
    batch_size = min(batch_size, n)

    pin_memory = model_device.type == "cuda"
    g = torch.Generator(device="cpu")
    g.manual_seed(int(random_state))
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=g,
        pin_memory=pin_memory,
    )
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(
                device=model_device,
                dtype=model_dtype,
                non_blocking=pin_memory,
            )
            yb = yb.to(
                device=model_device,
                dtype=model_dtype,
                non_blocking=pin_memory,
            )
            opt.zero_grad(set_to_none=True)
            outputs, _ = model(xb)
            loss = paper_mse_loss(outputs, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("ADMTSK training loss became non-finite.")
            loss.backward()
            opt.step()

    model.eval()
    model.is_fitted_ = True
    model.training_metadata = {
        "learning_rate": learning_rate,
        "batch_fraction": batch_fraction,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": "Adam",
        "scheduler": None,
        "loss": "paper_one_hot_mse",
        "preprocessing": "caller_supplied_minmax_0_1",
    }
    return model


class SklearnADMTSKWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-style wrapper matching repository wrapper conventions.

    Prediction
    ----------
    * ``out_features == 1``: sigmoid → threshold 0.5; proba columns [1-p, p]
    * ``out_features >= 2``: softmax / argmax (paper-faithful multiclass
      and two-output binary)

    Softmax probabilities on MSE-trained scores are uncalibrated
    compatibility scores.
    """

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
        if getattr(self.model, "is_fitted_", False) and getattr(
            self.model, "_classes", None
        ) is not None:
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
        learning_rate: float = 0.001,
        batch_fraction: float = 0.2,
        epochs: int = 50,
        random_state: int = 0,
    ):
        train_admtsk_model(
            self.model,
            X,
            y,
            learning_rate=learning_rate,
            batch_fraction=batch_fraction,
            epochs=epochs,
            random_state=random_state,
            device=self.device,
        )
        self.classes_ = np.asarray(self.model._classes)
        self.is_fitted_ = True
        return self

    def _iter_batches(self, X: Tensor):
        n = X.shape[0]
        bs = max(1, self.batch_size)
        for start in range(0, n, bs):
            yield X[start : start + bs]

    def _forward_logits(self, X) -> Tensor:
        X_t = self._convert_to_tensor(X)
        if X_t.shape[0] == 0:
            return torch.empty(
                (0, self.model.out_features),
                dtype=self.dtype,
            )

        pin_memory = (
            torch.device(self.device).type == "cuda"
            if not isinstance(self.device, torch.device)
            else self.device.type == "cuda"
        )
        was_training = self.model.training
        self.model.eval()
        chunks: List[Tensor] = []
        try:
            with torch.no_grad():
                for xb in self._iter_batches(X_t):
                    xb = xb.to(
                        device=self.device,
                        dtype=self.dtype,
                        non_blocking=pin_memory,
                    )
                    chunks.append(self.model(xb)[0].detach().cpu())
        finally:
            if was_training:
                self.model.train()
        return torch.cat(chunks, dim=0)

    def predict(self, X):
        logits = self._forward_logits(X)
        classes = getattr(self.model, "_classes", None)
        if logits.shape[0] == 0:
            if classes is not None:
                return np.empty((0,), dtype=np.asarray(classes).dtype)
            return np.empty((0,), dtype=np.int64)

        if self.model.out_features == 1:
            proba = torch.sigmoid(logits).squeeze(-1)
            pred = (proba >= 0.5).to(torch.long).cpu().numpy()
        else:
            pred = logits.argmax(dim=1).cpu().numpy()

        if classes is not None and len(classes) == self.model.out_features:
            return np.asarray(classes)[pred]
        if classes is not None and self.model.out_features == 1 and len(classes) == 2:
            return np.asarray(classes)[pred]
        return pred

    def predict_proba(self, X):
        logits = self._forward_logits(X)
        if logits.shape[0] == 0:
            n_cols = 2 if self.model.out_features == 1 else self.model.out_features
            return np.empty((0, n_cols), dtype=np.float32)

        if self.model.out_features == 1:
            p = torch.sigmoid(logits)
            if p.ndim == 1:
                p = p.unsqueeze(1)
            proba = torch.cat([1.0 - p, p], dim=1)
        else:
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()

    def decision_function(self, X):
        logits = self._forward_logits(X).detach().cpu().numpy()
        if logits.shape[1] == 1:
            return logits.ravel()
        return logits

    def score(self, X, y):
        return accuracy_score(np.asarray(y).reshape(-1), self.predict(X))

    def _convert_to_tensor(self, data):
        """Convert inputs to a CPU tensor; batches move to device later."""
        if isinstance(data, np.ndarray):
            return torch.as_tensor(data, dtype=self.dtype)
        if isinstance(data, torch.Tensor):
            return data.detach().to(device="cpu", dtype=self.dtype)
        if isinstance(data, pd.DataFrame):
            return torch.as_tensor(data.to_numpy(), dtype=self.dtype)
        raise ValueError("Input must be a NumPy array, DataFrame, or Tensor.")

    def get_params(self, deep=True):
        return {
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
        }

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
        return self


if __name__ == "__main__":
    D, R, C, B = 16, 3, 2, 8
    model = ADMTSK(D, R, C, binary=True, paper_mode=True, dtype=torch.float32)
    X = torch.rand(B, D)
    y, recon = model(X)
    print(f"lambda={float(model.lambda_value):.6f}")
    assert y.shape == (B, C)
    print("ok")
