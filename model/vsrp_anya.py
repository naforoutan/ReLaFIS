"""
VSRP-AnYa-EFS: Very Sparse Random Projection + AnYa Evolving Fuzzy System.

Reference
---------
H. Huang, H.-J. Rong, Z.-X. Yang, C.-M. Vong,
"Jointly evolving and compressing fuzzy system for feature reduction and
classification," Information Sciences, vol. 579, pp. 218–230, 2021.
doi: 10.1016/j.ins.2021.08.003

Status: paper-aligned independent reimplementation
(not official code; not claimed as exact numeric table reproduction).

Paper equations implemented
---------------------------
- AnYa data-cloud evolution: (1)–(8)
- VSRP / RSB compression and system output: (9)–(19)
- Local weighted RLS: (20)–(28)
- Algorithm 1 online learning procedure

Projection modes
----------------
- ``paper_dynamic`` (default, report.ipynb): one deterministic RSB matrix
  ``R_k`` per online training step, shared by all clouds for that sample.
  Held-out inference uses a **deterministic fixed inference projection
  adaptation** (one RSB matrix from ``inference_seed`` for the whole fitted
  model). The paper underspecifies test-time R handling; this adaptation is
  order- and batch-invariant and does not mutate training state.
- ``fixed_per_cloud_adaptation``: named project adaptation with one fixed R
  per cloud at creation time (not paper Algorithm 1).
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
from torch import Tensor, nn

from model.label_utils import label_key, stable_unique, validate_declared_classes

ArrayLike = Union[np.ndarray, torch.Tensor, pd.DataFrame]

PROJECTION_PAPER_DYNAMIC = "paper_dynamic"
PROJECTION_FIXED_PER_CLOUD = "fixed_per_cloud_adaptation"
_VALID_PROJECTION_MODES = (PROJECTION_PAPER_DYNAMIC, PROJECTION_FIXED_PER_CLOUD)


def projected_dimension(d_ext: int, compression_ratio: int) -> int:
    """``L = ceil((d + 1) / compression_ratio)`` with ``d_ext = d + 1``."""
    d_ext = int(d_ext)
    compression_ratio = int(compression_ratio)
    if d_ext < 1:
        raise ValueError(f"d_ext must be >= 1, got {d_ext}")
    if compression_ratio < 1:
        raise ValueError(f"compression_ratio must be >= 1, got {compression_ratio}")
    return max(1, int(math.ceil(d_ext / compression_ratio)))


def make_rsb_matrix_numpy(
    d_ext: int,
    proj_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """RSB matrix (paper eq. 11), shape ``(d_ext, L)``, float64.

    ``s = sqrt(d + 1)``; entries in ``{-sqrt(s), 0, +sqrt(s)}`` with
    probabilities ``1/(2s)``, ``1 - 1/s``, ``1/(2s)``.
    """
    d_ext = int(d_ext)
    proj_dim = int(proj_dim)
    if d_ext < 1 or proj_dim < 1:
        raise ValueError("d_ext and proj_dim must be >= 1")
    s = math.sqrt(d_ext)
    magnitude = math.sqrt(s)
    p = 1.0 / (2.0 * s)
    u = rng.random((d_ext, proj_dim))
    R = np.zeros((d_ext, proj_dim), dtype=np.float64)
    R[u < p] = -magnitude
    R[u > 1.0 - p] = magnitude
    if not np.isfinite(R).all():
        raise RuntimeError("RSB matrix contains non-finite values.")
    allowed = np.asarray([-magnitude, 0.0, magnitude], dtype=np.float64)
    flat = R.reshape(-1)
    ok = np.any(np.isclose(flat[:, None], allowed[None, :], rtol=0.0, atol=1e-12), axis=1)
    if not bool(ok.all()):
        raise RuntimeError("RSB matrix contains values outside {-√s, 0, +√s}.")
    return R


def _make_rsb_matrix(
    d_ext: int,
    proj_dim: int,
    generator: torch.Generator | None = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Torch RSB helper (eq. 11); prefers NumPy path when no generator."""
    if generator is not None:
        d_ext = int(d_ext)
        proj_dim = int(proj_dim)
        s = math.sqrt(d_ext)
        magnitude = math.sqrt(s)
        p = 1.0 / (2.0 * s)
        u = torch.rand(d_ext, proj_dim, generator=generator, device="cpu", dtype=torch.float64)
        R = torch.zeros(d_ext, proj_dim, dtype=torch.float64)
        R = torch.where(u < p, torch.full_like(R, -magnitude), R)
        R = torch.where(u > 1.0 - p, torch.full_like(R, magnitude), R)
        return R.to(device=device or "cpu", dtype=dtype or torch.float32)
    seed = 0
    rng = np.random.default_rng(seed)
    R = make_rsb_matrix_numpy(d_ext, proj_dim, rng)
    return torch.as_tensor(R, device=device or "cpu", dtype=dtype or torch.float32)


def local_wrls_step_numpy(
    C: np.ndarray,
    Q: np.ndarray,
    u: np.ndarray,
    lam: float,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """One local weighted RLS step (paper eqs. 24–25), float64 reference.

    Parameters
    ----------
    C : (L, L) covariance
    Q : (L, C) consequent
    u : (L,) compressed unweighted regressor ``(1/√L) Rᵀ x_e``
    lam : cloud normalized density λ_i
    target : (C,) one-hot / multi-output target
    """
    C = np.asarray(C, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    lam = float(lam)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError(f"lambda must be finite and >= 0, got {lam}")
    L = u.shape[0]
    if C.shape != (L, L):
        raise ValueError(f"C shape {C.shape} != {(L, L)}")
    if Q.shape[0] != L:
        raise ValueError(f"Q rows {Q.shape[0]} != L={L}")
    if not np.isfinite(C).all() or not np.isfinite(Q).all() or not np.isfinite(u).all():
        raise ValueError("Non-finite inputs to local wRLS.")
    if not np.isfinite(target).all():
        raise ValueError("Non-finite wRLS target.")

    Cu = C @ u
    denom = 1.0 + lam * float(u @ Cu)
    if not math.isfinite(denom) or denom <= 0.0:
        raise RuntimeError(f"Invalid wRLS denominator: {denom}")
    C_new = C - (lam * np.outer(Cu, Cu)) / denom
    C_new = 0.5 * (C_new + C_new.T)
    residual = target - (u @ Q)
    Q_new = Q + np.outer(C_new @ u, residual) * lam
    if not np.isfinite(C_new).all() or not np.isfinite(Q_new).all():
        raise RuntimeError("Non-finite local wRLS update.")
    return C_new, Q_new


class VSRPAnyaEFS(nn.Module):
    """Online VSRP-AnYa-EFS (custom-fit; no Adam / SGD / backprop).

    Constructor ``rules`` is only a compatibility hint (``rules_hint``).
    Cloud count evolves automatically unless ``max_rules`` is set explicitly.
    """

    N_LINGUISTIC_CATEGORIES = 4

    def __init__(
        self,
        in_features: int,
        rules: int,
        out_features: int,
        binary: bool = False,
        drop_out_p: float = 0.0,
        compression_ratio: int = 3,
        proj_dim: int | None = None,
        seed: int = 0,
        train_antecedents: bool = True,
        device=None,
        dtype=None,
        psi: float = 1.0,
        p_init: float = 500.0,
        sigma_eps: float = 1e-6,
        density_epsilon: float | None = None,
        max_rules: Optional[int] = None,
        projection_mode: str = PROJECTION_PAPER_DYNAMIC,
        inference_seed: int | None = None,
    ):
        super().__init__()
        factory_kwargs = self.factory_kwargs = {"device": device, "dtype": dtype}
        del train_antecedents

        in_features = int(in_features)
        rules = int(rules)
        out_features = int(out_features)
        drop_out_p = float(drop_out_p)
        compression_ratio = int(compression_ratio)
        psi = float(psi)
        p_init = float(p_init)
        if density_epsilon is not None:
            sigma_eps = float(density_epsilon)
        else:
            sigma_eps = float(sigma_eps)

        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if rules < 1:
            raise ValueError(f"rules must be >= 1, got {rules}")
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        if not (0.0 <= drop_out_p <= 1.0):
            raise ValueError(f"drop_out_p must be in [0, 1], got {drop_out_p}")
        if compression_ratio < 1:
            raise ValueError(f"compression_ratio must be >= 1, got {compression_ratio}")
        if not (0.0 < psi <= 1.0):
            raise ValueError(f"psi must be in (0, 1], got {psi}")
        if p_init <= 0.0:
            raise ValueError(f"p_init must be > 0, got {p_init}")
        if sigma_eps <= 0.0:
            raise ValueError(f"density_epsilon/sigma_eps must be > 0, got {sigma_eps}")
        if max_rules is not None and int(max_rules) < 1:
            raise ValueError(f"max_rules must be None or >= 1, got {max_rules}")
        if projection_mode not in _VALID_PROJECTION_MODES:
            raise ValueError(
                f"projection_mode must be one of {_VALID_PROJECTION_MODES}, "
                f"got {projection_mode!r}"
            )

        self.binary = bool(binary)
        self.in_features = in_features
        if self.binary:
            if out_features not in (1, 2):
                raise ValueError(
                    "Binary VSRP-AnYa-EFS requires out_features in {1, 2} "
                    f"(resolved to 2); got {out_features}."
                )
            self.out_features = 2
        else:
            self.out_features = out_features

        self.rules_hint = rules
        self.max_rules = int(max_rules) if max_rules is not None else None
        self.rules_count = 0
        self.drop_out_p = drop_out_p
        self.compression_ratio = compression_ratio
        self.device = device
        self._dtype = dtype
        self.seed = int(seed)
        self.inference_seed = (
            int(inference_seed)
            if inference_seed is not None
            else int(seed) + 1_000_003
        )
        # Unused compatibility parameter — does not affect paper equations.
        self.psi = psi
        self.unused_compatibility_parameter = "psi"
        self.p_init = p_init
        self.sigma_eps = sigma_eps
        self.density_epsilon = sigma_eps
        self.projection_mode = str(projection_mode)

        d_ext = in_features + 1
        proj_dim_was_explicit = proj_dim is not None
        if proj_dim is None:
            proj_dim = projected_dimension(d_ext, self.compression_ratio)
        else:
            proj_dim = int(proj_dim)
            if proj_dim < 1:
                raise ValueError(f"proj_dim must be >= 1, got {proj_dim}")
            if proj_dim > d_ext:
                raise ValueError(
                    f"proj_dim ({proj_dim}) cannot exceed d+1 ({d_ext}) "
                    "in paper-aligned VSRP-AnYa-EFS."
                )
        self.proj_dim = int(proj_dim)
        self.d_ext = d_ext
        self.proj_dim_was_explicit = bool(proj_dim_was_explicit)
        self.compression_source = (
            "explicit_proj_dim" if proj_dim_was_explicit else "compression_ratio"
        )

        # Protocol / Evaluator flags
        self.uses_custom_fit = True
        self.supports_backprop_training = False
        self.uses_generic_optimizer = False
        self.uses_reconstruction = False
        self.uses_reconstruction_loss = False
        self.uses_vsrp_anya_protocol = True
        self.uses_online_anya_evolution = True
        self.uses_local_fwrls = True
        self.required_input_scaling = "minmax_m1_1"
        self.training_mode = "online_anya_vsrp_local_wrls"
        self.paper_test_time_updates = False
        self.implementation_status = "paper_aligned_independent_reimplementation"
        self.rsb_distribution = "very_sparse_bernoulli"
        self.rsb_scaling = "1_over_sqrt_L"
        self.is_fitted_ = False

        # Local means Γ_i
        self.register_buffer("centers", torch.zeros(0, in_features, **factory_kwargs))
        # Focal points ξ_i (distinct from Γ)
        self.register_buffer("foci", torch.zeros(0, in_features, **factory_kwargs))
        self.register_buffer("radii", torch.zeros(0, **factory_kwargs))
        self.register_buffer("scatters", torch.zeros(0, **factory_kwargs))
        self.register_buffer("supports", torch.zeros(0, **factory_kwargs))
        self.register_buffer("avg_sq", torch.zeros(0, **factory_kwargs))
        # Used only in fixed_per_cloud_adaptation; empty in paper_dynamic.
        self.register_buffer(
            "R", torch.zeros(0, d_ext, self.proj_dim, **factory_kwargs)
        )
        self.consequent = nn.Parameter(
            torch.zeros(0, self.proj_dim, self.out_features, **factory_kwargs),
            requires_grad=False,
        )
        # Inactive compatibility stubs (never used in forward/fit).
        self.decoder_linear = nn.Linear(
            in_features=max(self.rules_hint, 1),
            out_features=in_features,
            bias=True,
            **factory_kwargs,
        )
        for p in self.decoder_linear.parameters():
            p.requires_grad_(False)
        self.drop_out = nn.Dropout(p=drop_out_p)

        self._next_rule_index = 0
        self._train_step_index = 0
        self._rng_state: Optional[Dict[str, Any]] = None
        self._P: List[np.ndarray] = []
        self._classes: Optional[np.ndarray] = None
        self._n_seen: int = 0
        self._global_mean: Optional[np.ndarray] = None
        self._global_avg_sq: float = 0.0
        self._cloud_updated_flags: List[bool] = []
        self._checkpoint_load_strict: bool = True
        self._inference_R: Optional[np.ndarray] = None
        self.inference_projection_adaptation = (
            "deterministic_fixed_inference_projection"
        )

    @property
    def model_metadata(self) -> Dict[str, Any]:
        adaptations = [
            "sklearn probability-like compatibility scores (softmax of LS scores)",
            "train-only MinMaxScaler feature_range=(-1, 1)",
            "deterministic fixed inference projection adaptation "
            "(one RSB from inference_seed; paper underspecifies test-time R)",
        ]
        if self.max_rules is not None:
            adaptations.append(
                f"explicit max_rules={self.max_rules} cloud cap (project adaptation)"
            )
        if self.projection_mode == PROJECTION_FIXED_PER_CLOUD:
            adaptations.append("fixed_per_cloud_adaptation projection mode")
        return {
            "model_name": "VSRP-AnYa-EFS",
            "implementation_status": self.implementation_status,
            "training_mode": self.training_mode,
            "required_input_scaling": self.required_input_scaling,
            "compression_factor": self.compression_ratio,
            "compression_source": self.compression_source,
            "projected_dimension": self.proj_dim,
            "extended_input_dimension": self.d_ext,
            "rules_hint": self.rules_hint,
            "max_rules": self.max_rules,
            "p_init": self.p_init,
            "projection_mode": self.projection_mode,
            "inference_projection_adaptation": self.inference_projection_adaptation,
            "inference_seed": int(self.inference_seed),
            "rsb_distribution": self.rsb_distribution,
            "rsb_scaling": self.rsb_scaling,
            "test_time_updates": self.paper_test_time_updates,
            "adaptations": adaptations,
        }

    @property
    def mean(self) -> torch.Tensor:
        if self.centers.numel() == 0:
            kw = {"device": self.centers.device, "dtype": self.centers.dtype}
            return torch.zeros(self.in_features, 1, **kw)
        return self.centers.transpose(0, 1)

    def local_density(self, X: torch.Tensor) -> torch.Tensor:
        """Local density vs cloud means Γ (eq. 12), shape (batch, rules)."""
        if self.rules_count < 1:
            return X.new_zeros(X.shape[0], 0)
        diff = X.unsqueeze(1) - self.centers.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        center_norm2 = (self.centers ** 2).sum(dim=-1)
        sigma2 = (self.avg_sq - center_norm2).clamp(min=self.density_epsilon)
        return 1.0 / (1.0 + dist2 / sigma2.unsqueeze(0))

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """Normalized local densities λ_i (eq. 13)."""
        self._require_fitted()
        c = self.local_density(X)
        if c.shape[1] == 0:
            raise RuntimeError("Cannot encode with zero rules.")
        firing = F.normalize(c, p=1, dim=1)
        if not torch.isfinite(firing).all():
            raise ValueError("Normalized firing is not finite.")
        if (firing < -1e-8).any():
            raise ValueError("Normalized firing must be nonnegative.")
        row = firing.sum(dim=1)
        if not torch.allclose(row, torch.ones_like(row), atol=1e-4):
            raise RuntimeError("Normalized firing rows must sum to one.")
        return firing

    def _xe(self, X: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
        return torch.cat([ones, X], dim=1)

    def compressed_antecedents(
        self, X: torch.Tensor, firing: torch.Tensor, R: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``g_i = λ_i * u`` with ``u = (1/√L) x_e R`` (eqs. 14–15)."""
        xe = self._xe(X)
        scale = 1.0 / math.sqrt(self.proj_dim)
        if self.projection_mode == PROJECTION_PAPER_DYNAMIC:
            if R is None:
                raise ValueError("paper_dynamic compressed_antecedents requires R.")
            # R: (d_ext, L) shared; u: (B, L); g: (B, R, L)
            u = scale * (xe @ R)
            return firing.unsqueeze(-1) * u.unsqueeze(1)
        # fixed per cloud: self.R is (rules, d_ext, L)
        u = scale * torch.einsum("bd,rdl->brl", xe, self.R)
        return firing.unsqueeze(-1) * u

    def consequent_output(self, g: torch.Tensor) -> torch.Tensor:
        """``ŷ = Σ_i g_i Q_i`` (eqs. 16–18)."""
        return torch.einsum("brl,rlo->bo", g, self.consequent)

    def linguistic_richness(self, per_rule: bool = False):
        """Not applicable — VSRP-AnYa-EFS has no shared 4-category vocabulary."""
        r = max(self.rules_count, 1)
        with torch.no_grad():
            entropies = torch.full(
                (r,), float("nan"), dtype=torch.float64, device=self.centers.device
            )
        if per_rule:
            return float("nan"), entropies
        return float("nan")

    def relaxation_rate(self, per_rule: bool = False):
        """Not applicable — no learned relaxation gate."""
        r = max(self.rules_count, 1)
        with torch.no_grad():
            rates = torch.full(
                (r,), float("nan"), dtype=torch.float64, device=self.centers.device
            )
        if per_rule:
            return float("nan"), rates
        return float("nan")

    def forward(self, X: torch.Tensor):
        self._require_fitted()
        if X.ndim != 2 or X.shape[1] != self.in_features:
            raise ValueError(
                f"Expected X shape (N, {self.in_features}), got {tuple(X.shape)}"
            )
        if not torch.isfinite(X).all():
            raise ValueError("X contains NaN or Inf.")
        firing = self.encode(X)
        if self.projection_mode == PROJECTION_PAPER_DYNAMIC:
            # Deterministic fixed inference projection adaptation:
            # one RSB matrix for the entire fitted model (order/batch invariant).
            R_t = torch.as_tensor(
                self._get_inference_R(), device=X.device, dtype=X.dtype
            )
            g = self.compressed_antecedents(X, firing, R=R_t)
            y = self.consequent_output(g)
        else:
            g = self.compressed_antecedents(X, firing)
            y = self.consequent_output(g)
        if not torch.isfinite(y).all():
            raise RuntimeError("Non-finite VSRP-AnYa-EFS outputs.")
        return y, X.detach()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, y: ArrayLike):
        X_np, y_np = self._to_numpy_xy(X, y)
        classes = stable_unique(y_np)
        self._validate_class_schema(classes, require_complete=True)
        self._classes = classes
        if not self.binary and len(classes) != self.out_features:
            raise ValueError(
                f"out_features={self.out_features} but training has "
                f"{len(classes)} classes."
            )
        self._reset_state()
        self._classes = classes
        # Algorithm 1: k=1 initializes the first cloud only (no projection / wRLS).
        self._bootstrap_first_cloud(X_np[0])
        for i in range(1, X_np.shape[0]):
            self._online_step(X_np[i], y_np[i])
        self._finalize_fit()
        return self

    def partial_fit(
        self, X: ArrayLike, y: ArrayLike, classes: Optional[ArrayLike] = None
    ):
        X_np, y_np = self._to_numpy_xy(X, y)
        if self._classes is None:
            if classes is None:
                raise ValueError(
                    "partial_fit requires classes= on the first call "
                    "(sklearn-style complete schema)."
                )
            declared = validate_declared_classes(classes)
            self._validate_class_schema(declared, require_complete=True)
            self._classes = declared
            if not self.binary and len(declared) != self.out_features:
                raise ValueError(
                    f"Declared {len(declared)} classes but out_features={self.out_features}."
                )
            self._reset_state()
            self._classes = declared
        else:
            if classes is not None:
                declared = validate_declared_classes(classes)
                existing_keys = [label_key(c) for c in np.asarray(self._classes)]
                declared_keys = [label_key(c) for c in declared]
                if declared_keys != existing_keys:
                    # Allow identical multiset only if same order.
                    if set(declared_keys) != set(existing_keys) or declared_keys != existing_keys:
                        raise ValueError(
                            "classes= must match the established schema order; "
                            f"existing={existing_keys!r}, declared={declared_keys!r}"
                        )
            self._reject_unseen_labels(y_np)

        start = 0
        if self.rules_count < 1:
            # First call / empty state: Algorithm 1 k=1 bootstrap (no wRLS).
            self._reject_unseen_labels(y_np[:1])
            self._bootstrap_first_cloud(X_np[0])
            start = 1
        for i in range(start, X_np.shape[0]):
            self._online_step(X_np[i], y_np[i])
        self._finalize_fit()
        return self

    def _finalize_fit(self) -> None:
        self._sync_torch_state()
        if self.rules_count < 1:
            raise RuntimeError("Fitting produced zero rules.")
        if self._classes is None or len(self._classes) < 1:
            raise RuntimeError("Fitting left class schema incomplete.")
        for P in self._P:
            if not np.isfinite(P).all():
                raise RuntimeError("Covariance matrix is not finite after fit.")
        if not torch.isfinite(self.consequent).all():
            raise RuntimeError("Consequent Q is not finite after fit.")
        self.is_fitted_ = True

    def _validate_class_schema(self, classes: np.ndarray, *, require_complete: bool) -> None:
        classes = np.asarray(classes).reshape(-1)
        if require_complete and len(classes) < 1:
            raise ValueError("Class schema is empty.")
        keys = [label_key(c) for c in classes]
        if len(keys) != len(set(keys)):
            raise ValueError("Class schema contains duplicate labels.")
        if self.binary and len(classes) != 2:
            raise ValueError("Binary VSRP-AnYa-EFS requires exactly two classes.")
        if (not self.binary) and require_complete and len(classes) != self.out_features:
            raise ValueError(
                f"Multiclass schema has {len(classes)} classes but "
                f"out_features={self.out_features}."
            )

    def _reject_unseen_labels(self, y_np: np.ndarray) -> None:
        assert self._classes is not None
        known = {label_key(c) for c in np.asarray(self._classes)}
        for v in np.asarray(y_np).reshape(-1):
            if label_key(v) not in known:
                raise ValueError(
                    f"Unseen class {label_key(v)!r} after schema establishment; "
                    "refusing to change output dimensions / Q / C."
                )

    def _require_fitted(self) -> None:
        if not self.is_fitted_ or self.rules_count < 1:
            raise RuntimeError(
                "VSRPAnyaEFS is not fitted. Call fit(X, y) before forward/encode."
            )

    def _factory_device_dtype(self) -> Dict[str, Any]:
        return {
            "device": self.centers.device,
            "dtype": self.centers.dtype if self.centers.dtype is not None else torch.float32,
        }

    def _to_numpy_xy(self, X: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float64, copy=False)
        elif isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y_np = y.values
        elif isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)
        y_np = y_np.reshape(-1)

        if X_np.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X_np.shape}")
        if X_np.shape[0] < 1:
            raise ValueError("X must contain at least one sample.")
        if y_np.shape[0] < 1:
            raise ValueError("y must not be empty.")
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X_np.shape[0]} vs {y_np.shape[0]}"
            )
        if X_np.shape[1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} features, got {X_np.shape[1]}"
            )
        if not np.isfinite(X_np).all():
            raise ValueError("X contains NaN or infinite values.")
        if y_np.dtype.kind in "iufc":
            if not np.isfinite(y_np.astype(np.float64, copy=False)).all():
                raise ValueError("y contains NaN or infinite labels.")
        return X_np, y_np

    def _reset_state(self) -> None:
        self.rules_count = 0
        self._P = []
        self._n_seen = 0
        self._global_mean = None
        self._global_avg_sq = 0.0
        self.is_fitted_ = False
        self._next_rule_index = 0
        self._train_step_index = 0
        self._cloud_updated_flags = []
        kw = self._factory_device_dtype()
        self.register_buffer("centers", torch.zeros(0, self.in_features, **kw))
        self.register_buffer("foci", torch.zeros(0, self.in_features, **kw))
        self.register_buffer("radii", torch.zeros(0, **kw))
        self.register_buffer("scatters", torch.zeros(0, **kw))
        self.register_buffer("supports", torch.zeros(0, **kw))
        self.register_buffer("avg_sq", torch.zeros(0, **kw))
        self.register_buffer("R", torch.zeros(0, self.d_ext, self.proj_dim, **kw))
        self.consequent = nn.Parameter(
            torch.zeros(0, self.proj_dim, self.out_features, **kw),
            requires_grad=False,
        )
        self._inference_R = None

    def _bootstrap_first_cloud(self, x: np.ndarray) -> None:
        """Algorithm 1 k=1: create first cloud; no projection / wRLS."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.in_features:
            raise ValueError(
                f"Sample has {x.shape[0]} features, expected {self.in_features}"
            )
        if not np.isfinite(x).all():
            raise ValueError("Sample contains NaN or infinite values.")
        if self.rules_count != 0:
            raise RuntimeError("First-cloud bootstrap requires empty rule set.")
        self._update_global(x)
        self._add_cloud(x)
        # Leave Q=0, C=p0*I, rho=1; do not advance train_step / run wRLS.
        self._inference_R = None

    def _get_inference_R(self) -> np.ndarray:
        """Fixed inference RSB (deterministic fixed inference projection adaptation)."""
        if self._inference_R is None:
            self._inference_R = self._rsb_from_seed(int(self.inference_seed))
        return self._inference_R

    def _target_vector(self, y_i) -> np.ndarray:
        if self._classes is None:
            raise ValueError("Class schema is not established.")
        class_to_index = {
            label_key(label): index
            for index, label in enumerate(np.asarray(self._classes).reshape(-1))
        }
        key = label_key(y_i)
        if key not in class_to_index:
            raise ValueError(f"Unknown class label: {key!r}")
        t = np.zeros(self.out_features, dtype=np.float64)
        t[class_to_index[key]] = 1.0
        return t

    def _global_density(self, v: np.ndarray) -> float:
        assert self._global_mean is not None
        diff = v - self._global_mean
        dist2 = float(np.dot(diff, diff))
        center_n2 = float(np.dot(self._global_mean, self._global_mean))
        sigma2 = max(self._global_avg_sq - center_n2, self.density_epsilon)
        return 1.0 / (1.0 + dist2 / sigma2)

    def _local_density_np(self, x: np.ndarray, i: int) -> float:
        g = self.centers[i].detach().cpu().numpy()
        diff = x - g
        dist2 = float(np.dot(diff, diff))
        avg_sq = float(self.avg_sq[i].detach().cpu().item())
        center_n2 = float(np.dot(g, g))
        sigma2 = max(avg_sq - center_n2, self.density_epsilon)
        return 1.0 / (1.0 + dist2 / sigma2)

    def _varrho(self, i: int) -> float:
        g = self.centers[i].detach().cpu().numpy()
        avg_sq = float(self.avg_sq[i].detach().cpu().item())
        return math.sqrt(max(avg_sq - float(np.dot(g, g)), self.density_epsilon))

    def _update_radius_eq7(self, i: int) -> None:
        """ρ ← ½(ρ_prev + ϱ) once for an updated existing cloud."""
        rho_prev = float(self.radii[i].detach().cpu().item())
        varrho = self._varrho(i)
        rho_new = 0.5 * (rho_prev + varrho)
        with torch.no_grad():
            self.radii[i] = rho_new
            self.scatters[i] = varrho

    def _update_global(self, x: np.ndarray) -> None:
        self._n_seen += 1
        n = self._n_seen
        x_n2 = float(np.dot(x, x))
        if self._global_mean is None:
            self._global_mean = x.copy()
            self._global_avg_sq = x_n2
        else:
            self._global_mean = ((n - 1) / n) * self._global_mean + x / n
            self._global_avg_sq = ((n - 1) / n) * self._global_avg_sq + x_n2 / n

    def _seed_for_train_step(self, step_index: int) -> int:
        # Deterministic, distinct from inference seeds.
        return int(self.seed) + 1_000_003 * int(step_index) + 17

    def _seed_for_inference_index(self, sample_index: int) -> int:
        """Deprecated compatibility alias.

        Held-out inference uses one fixed RSB from ``inference_seed``
        (deterministic fixed inference projection adaptation), not a
        per-row seed. Prefer ``_get_inference_R()``.
        """
        del sample_index
        return int(self.inference_seed)

    def _seed_for_cloud(self, creation_index: int) -> int:
        return int(self.seed) + int(creation_index)

    def _rsb_from_seed(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        return make_rsb_matrix_numpy(self.d_ext, self.proj_dim, rng)

    def _rsb_for_train_step(self, step_index: int) -> np.ndarray:
        return self._rsb_from_seed(self._seed_for_train_step(step_index))

    def _rsb_for_inference_index(self, sample_index: int) -> np.ndarray:
        """Deprecated alias: held-out inference uses one fixed matrix."""
        del sample_index
        return self._get_inference_R()

    def _new_fixed_cloud_rsb(self) -> torch.Tensor:
        creation_index = int(self._next_rule_index)
        self._next_rule_index += 1
        R = self._rsb_from_seed(self._seed_for_cloud(creation_index))
        kw = self._factory_device_dtype()
        return torch.as_tensor(R, **kw)

    def _add_cloud(self, x: np.ndarray) -> None:
        """Create a new cloud with ρ=1; do not apply radius recursion yet."""
        kw = self._factory_device_dtype()
        x_t = torch.as_tensor(x, **kw).unsqueeze(0)
        x_n2 = float(np.dot(x, x))
        Q_new = torch.zeros(1, self.proj_dim, self.out_features, **kw)
        rho0 = torch.ones(1, **kw)
        v0 = torch.full((1,), math.sqrt(self.density_epsilon), **kw)

        if self.projection_mode == PROJECTION_FIXED_PER_CLOUD:
            R_new = self._new_fixed_cloud_rsb().unsqueeze(0)
        else:
            # Keep buffer aligned but unused for projection.
            R_new = torch.zeros(1, self.d_ext, self.proj_dim, **kw)
            self._next_rule_index += 1

        if self.rules_count == 0:
            self.register_buffer("centers", x_t.clone())
            self.register_buffer("foci", x_t.clone())
            self.register_buffer("radii", rho0.clone())
            self.register_buffer("scatters", v0.clone())
            self.register_buffer("supports", torch.ones(1, **kw))
            self.register_buffer("avg_sq", torch.tensor([x_n2], **kw))
            self.register_buffer("R", R_new)
            self.consequent = nn.Parameter(Q_new, requires_grad=False)
            self._cloud_updated_flags = [False]
        else:
            self.register_buffer("centers", torch.cat([self.centers, x_t], dim=0))
            self.register_buffer("foci", torch.cat([self.foci, x_t], dim=0))
            self.register_buffer("radii", torch.cat([self.radii, rho0], dim=0))
            self.register_buffer("scatters", torch.cat([self.scatters, v0], dim=0))
            self.register_buffer(
                "supports", torch.cat([self.supports, torch.ones(1, **kw)], dim=0)
            )
            self.register_buffer(
                "avg_sq",
                torch.cat([self.avg_sq, torch.tensor([x_n2], **kw)], dim=0),
            )
            self.register_buffer("R", torch.cat([self.R, R_new], dim=0))
            self.consequent = nn.Parameter(
                torch.cat([self.consequent.data, Q_new], dim=0),
                requires_grad=False,
            )
            self._cloud_updated_flags.append(False)

        self._P.append(self.p_init * np.eye(self.proj_dim, dtype=np.float64))
        self.rules_count = int(self.centers.shape[0])
        # New cloud retains ρ = 1 until a later sample updates it.

    def _update_cloud(self, i: int, x: np.ndarray) -> None:
        m = float(self.supports[i].detach().cpu().item())
        c = self.centers[i].detach().cpu().numpy()
        avg_sq = float(self.avg_sq[i].detach().cpu().item())
        m_new = m + 1.0
        c_new = (m / m_new) * c + x / m_new
        x_n2 = float(np.dot(x, x))
        avg_sq_new = (m / m_new) * avg_sq + x_n2 / m_new
        kw = self._factory_device_dtype()
        with torch.no_grad():
            self.centers[i] = torch.as_tensor(c_new, **kw)
            self.supports[i] = m_new
            self.avg_sq[i] = avg_sq_new
            self.foci[i] = torch.as_tensor(x, **kw)
        self._update_radius_eq7(i)
        self._cloud_updated_flags[i] = True

    def _should_add_cloud(self, x: np.ndarray) -> bool:
        if self.rules_count < 1:
            return True
        if self.max_rules is not None and self.rules_count >= self.max_rules:
            return False

        dens_x = self._global_density(x)
        means = self.centers.detach().cpu().numpy()
        dens_means = np.array(
            [self._global_density(means[i]) for i in range(self.rules_count)]
        )
        density_extreme = bool(
            np.all(dens_x > dens_means + 1e-15) or np.all(dens_x < dens_means - 1e-15)
        )
        foci = self.foci.detach().cpu().numpy()
        dists = np.linalg.norm(foci - x.reshape(1, -1), axis=1)
        radii = self.radii.detach().cpu().numpy()
        far_from_all = bool(np.all(dists > radii + 1e-15))
        return density_extreme or far_from_all

    def _firing_np(self, x: np.ndarray) -> np.ndarray:
        if self.rules_count < 1:
            return np.zeros(0, dtype=np.float64)
        c = np.array(
            [self._local_density_np(x, i) for i in range(self.rules_count)],
            dtype=np.float64,
        )
        if not np.isfinite(c).all() or (c < 0).any():
            raise RuntimeError("Local densities became numerically unusable.")
        s = float(c.sum())
        if not math.isfinite(s) or s <= 0.0:
            raise RuntimeError("Local density sum is non-positive.")
        lam = c / s
        if not np.isfinite(lam).all() or (lam < -1e-12).any():
            raise RuntimeError("Normalized densities invalid.")
        if not np.isclose(lam.sum(), 1.0, atol=1e-8):
            raise RuntimeError("Normalized densities must sum to one.")
        return lam

    def _project_u_shared(self, x: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Shared ``u = (1/√L) Rᵀ x_e`` for all clouds (paper_dynamic)."""
        xe = np.concatenate([[1.0], x]).astype(np.float64)
        scale = 1.0 / math.sqrt(self.proj_dim)
        u = scale * (R.T @ xe)
        return np.tile(u.reshape(1, -1), (self.rules_count, 1))

    def _project_u_per_cloud(self, x: np.ndarray) -> np.ndarray:
        xe = np.concatenate([[1.0], x]).astype(np.float64)
        R = self.R.detach().cpu().numpy().astype(np.float64)
        scale = 1.0 / math.sqrt(self.proj_dim)
        return scale * np.einsum("rdl,d->rl", R, xe)

    def _fwrls_update(self, lam: np.ndarray, U: np.ndarray, target: np.ndarray) -> None:
        Q = self.consequent.detach().cpu().numpy().astype(np.float64)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(self._P) != self.rules_count:
            raise RuntimeError("Covariance count does not match rules_count.")
        for i in range(self.rules_count):
            C_new, Q_new = local_wrls_step_numpy(
                self._P[i], Q[i], U[i], float(lam[i]), target
            )
            self._P[i] = C_new
            Q[i] = Q_new
        kw = self._factory_device_dtype()
        with torch.no_grad():
            self.consequent.data = torch.as_tensor(Q, **kw)

    def _online_step(self, x: np.ndarray, y_i) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.in_features:
            raise ValueError(
                f"Sample has {x.shape[0]} features, expected {self.in_features}"
            )
        if not np.isfinite(x).all():
            raise ValueError("Sample contains NaN or infinite values.")
        self._reject_unseen_labels(np.asarray([y_i]))
        target = self._target_vector(y_i)
        self._update_global(x)

        if self.rules_count < 1:
            raise RuntimeError(
                "Online step requires an initialized first cloud; "
                "call fit/partial_fit bootstrap first."
            )
        if self._should_add_cloud(x):
            self._add_cloud(x)
        else:
            foci = self.foci.detach().cpu().numpy()
            j = int(np.argmin(np.linalg.norm(foci - x.reshape(1, -1), axis=1)))
            self._update_cloud(j, x)

        lam = self._firing_np(x)
        step = int(self._train_step_index)
        self._train_step_index += 1
        if self.projection_mode == PROJECTION_PAPER_DYNAMIC:
            R = self._rsb_for_train_step(step)
            U = self._project_u_shared(x, R)
        else:
            U = self._project_u_per_cloud(x)
        self._fwrls_update(lam, U, target)

    def _sync_torch_state(self) -> None:
        # Buffers already live on device; validate consistency only.
        self._validate_live_state(strict=True)

    def _validate_live_state(self, *, strict: bool) -> None:
        r = self.rules_count
        if r < 1:
            return
        if tuple(self.centers.shape) != (r, self.in_features):
            raise ValueError("centers shape mismatch.")
        if tuple(self.foci.shape) != (r, self.in_features):
            raise ValueError("foci shape mismatch.")
        if int(self.radii.shape[0]) != r or int(self.supports.shape[0]) != r:
            raise ValueError("radii/supports length mismatch.")
        if int(self.scatters.shape[0]) != r or int(self.avg_sq.shape[0]) != r:
            raise ValueError("scatters/avg_sq length mismatch.")
        if tuple(self.consequent.shape) != (r, self.proj_dim, self.out_features):
            raise ValueError("consequent shape mismatch.")
        if len(self._P) != r:
            raise ValueError("P count mismatch.")
        if len(self._cloud_updated_flags) != r:
            raise ValueError("cloud_updated_flags length mismatch.")
        for i, P in enumerate(self._P):
            if P.shape != (self.proj_dim, self.proj_dim):
                raise ValueError(f"P[{i}] shape mismatch.")
            if not np.isfinite(P).all():
                raise ValueError(f"P[{i}] is not finite.")
            if not np.allclose(P, P.T, atol=1e-5):
                raise ValueError(f"P[{i}] is not symmetric.")
        if self._classes is None:
            raise ValueError("Fitted model missing classes.")
        if self.binary and len(self._classes) != 2:
            raise ValueError("Binary fitted model must have exactly two classes.")
        if (not self.binary) and len(self._classes) != self.out_features:
            raise ValueError("Multiclass class count must equal out_features.")
        for name, t in (
            ("centers", self.centers),
            ("foci", self.foci),
            ("radii", self.radii),
            ("scatters", self.scatters),
            ("supports", self.supports),
            ("avg_sq", self.avg_sq),
            ("consequent", self.consequent),
        ):
            if not torch.isfinite(t).all():
                raise ValueError(f"{name} contains non-finite values.")
        if (self.supports <= 0).any():
            raise ValueError("supports must be positive.")
        if (self.radii < 0).any():
            raise ValueError("radii must be non-negative.")
        if (self.scatters < 0).any():
            raise ValueError("scatters must be non-negative.")
        if self._n_seen < 1:
            raise ValueError("n_seen must be >= 1 for a live cloud state.")
        if self._global_mean is None:
            raise ValueError("global_mean missing.")
        if not np.isfinite(self._global_mean).all():
            raise ValueError("global_mean is not finite.")
        if not math.isfinite(self._global_avg_sq):
            raise ValueError("global_avg_sq is not finite.")
        if self._inference_R is not None:
            if self._inference_R.shape != (self.d_ext, self.proj_dim):
                raise ValueError("inference_R shape mismatch.")
            if not np.isfinite(self._inference_R).all():
                raise ValueError("inference_R is not finite.")
        if strict and self.projection_mode == PROJECTION_FIXED_PER_CLOUD:
            if tuple(self.R.shape) != (r, self.d_ext, self.proj_dim):
                raise ValueError("R shape mismatch.")

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "rules_count": int(self.rules_count),
            "centers": self.centers.detach().cpu().numpy().copy(),
            "foci": self.foci.detach().cpu().numpy().copy(),
            "radii": self.radii.detach().cpu().numpy().copy(),
            "scatters": self.scatters.detach().cpu().numpy().copy(),
            "supports": self.supports.detach().cpu().numpy().copy(),
            "avg_sq": self.avg_sq.detach().cpu().numpy().copy(),
            "R": self.R.detach().cpu().numpy().copy(),
            "Q": self.consequent.detach().cpu().numpy().copy(),
            "P": [p.copy() for p in self._P],
            "n_seen": int(self._n_seen),
            "global_mean": None if self._global_mean is None else self._global_mean.copy(),
            "global_avg_sq": float(self._global_avg_sq),
            "classes": None if self._classes is None else np.asarray(self._classes).copy(),
            "next_rule_index": int(self._next_rule_index),
            "train_step_index": int(self._train_step_index),
            "projection_mode": self.projection_mode,
            "inference_seed": int(self.inference_seed),
            "inference_R": None
            if self._inference_R is None
            else self._inference_R.copy(),
            "cloud_updated_flags": list(self._cloud_updated_flags),
        }

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "P": [p.copy() for p in self._P],
            "classes": None if self._classes is None else np.asarray(self._classes).tolist(),
            "is_fitted": bool(self.is_fitted_),
            "rules_count": int(self.rules_count),
            "rules_hint": int(self.rules_hint),
            "out_features": int(self.out_features),
            "binary": bool(self.binary),
            "in_features": int(self.in_features),
            "compression_ratio": int(self.compression_ratio),
            "proj_dim": int(self.proj_dim),
            "d_ext": int(self.d_ext),
            "seed": int(self.seed),
            "inference_seed": int(self.inference_seed),
            "inference_R": None
            if self._inference_R is None
            else self._inference_R.copy(),
            "inference_projection_adaptation": self.inference_projection_adaptation,
            "next_rule_index": int(self._next_rule_index),
            "train_step_index": int(self._train_step_index),
            "psi": float(self.psi),
            "unused_compatibility_parameter": self.unused_compatibility_parameter,
            "p_init": float(self.p_init),
            "sigma_eps": float(self.sigma_eps),
            "density_epsilon": float(self.density_epsilon),
            "max_rules": self.max_rules,
            "n_seen": int(self._n_seen),
            "global_mean": None if self._global_mean is None else self._global_mean.copy(),
            "global_avg_sq": float(self._global_avg_sq),
            "compression_source": self.compression_source,
            "implementation_status": self.implementation_status,
            "training_mode": self.training_mode,
            "projection_mode": self.projection_mode,
            "required_input_scaling": self.required_input_scaling,
            "cloud_updated_flags": list(self._cloud_updated_flags),
        }

    def _resolve_checkpoint_strict(self, strict: Optional[bool] = None) -> bool:
        if strict is not None:
            return bool(strict)
        return bool(getattr(self, "_checkpoint_load_strict", True))

    # Strict tensor keys required in a complete fitted checkpoint.
    _STRICT_REQUIRED_TENSOR_KEYS = (
        "centers",
        "foci",
        "radii",
        "scatters",
        "supports",
        "avg_sq",
        "R",
        "consequent",
        "_extra_state",
    )
    _COMPAT_MODULE_KEYS = frozenset(
        {"decoder_linear.weight", "decoder_linear.bias"}
    )
    _STRICT_REQUIRED_EXTRA_KEYS = (
        "classes",
        "P",
        "is_fitted",
        "rules_count",
        "rules_hint",
        "in_features",
        "out_features",
        "binary",
        "proj_dim",
        "projection_mode",
        "inference_seed",
        "n_seen",
        "train_step_index",
        "next_rule_index",
        "global_mean",
        "global_avg_sq",
        "cloud_updated_flags",
        "p_init",
        "max_rules",
        "compression_ratio",
        "compression_source",
        "seed",
        "sigma_eps",
        "density_epsilon",
    )

    def _preflight_validate_checkpoint(
        self,
        state_dict: Dict[str, Any],
        extra: Optional[Dict[str, Any]],
        *,
        strict: bool,
    ) -> None:
        """Validate checkpoint tensors/metadata without mutating model state."""
        # ---- key-set validation (strict) ----
        present_keys = set(state_dict.keys())
        allowed_keys = set(self._STRICT_REQUIRED_TENSOR_KEYS) | set(
            self._COMPAT_MODULE_KEYS
        )
        # Current model keys (buffers/params) are always allowed.
        allowed_keys |= set(self.state_dict().keys())
        unexpected = sorted(present_keys - allowed_keys)
        if strict and unexpected:
            raise ValueError(
                f"Strict checkpoint has unexpected keys: {unexpected}"
            )

        required_tensors = (
            "centers",
            "foci",
            "radii",
            "scatters",
            "supports",
            "avg_sq",
            "R",
            "consequent",
        )
        if strict:
            missing_tensors = [k for k in required_tensors if k not in state_dict]
            if missing_tensors:
                raise ValueError(
                    f"Strict checkpoint missing required tensors: {missing_tensors}"
                )
            if "_extra_state" not in state_dict and extra is None:
                raise ValueError(
                    "Strict VSRP checkpoint is missing required _extra_state."
                )
        else:
            if "centers" not in state_dict:
                raise ValueError("Checkpoint missing centers.")

        if extra is None:
            if strict:
                raise ValueError(
                    "Strict VSRP checkpoint is missing required _extra_state."
                )
            warnings.warn(
                "Legacy tensor-only VSRP checkpoint has no extra state; "
                "the model will remain unfitted.",
                RuntimeWarning,
            )
            # Still validate available tensors below (non-strict path).
        else:
            if strict:
                missing_extra = [
                    k for k in self._STRICT_REQUIRED_EXTRA_KEYS if k not in extra
                ]
                if missing_extra:
                    raise ValueError(
                        f"Strict checkpoint missing extra keys: {missing_extra}"
                    )

        centers = state_dict.get("centers")
        if centers is None or not torch.is_tensor(centers) or centers.ndim != 2:
            raise ValueError("Checkpoint centers must be a rank-2 tensor.")
        r, d = int(centers.shape[0]), int(centers.shape[1])
        if d != self.in_features:
            raise ValueError(
                f"Checkpoint in_features {d} != model {self.in_features}"
            )
        if r < 1 and strict:
            raise ValueError("Strict checkpoint has zero rules.")

        consequent = state_dict.get("consequent")
        if consequent is None or not torch.is_tensor(consequent) or consequent.ndim != 3:
            raise ValueError("Checkpoint consequent must be a rank-3 tensor.")
        rq, L, o = (
            int(consequent.shape[0]),
            int(consequent.shape[1]),
            int(consequent.shape[2]),
        )
        if rq != r:
            raise ValueError("Checkpoint consequent rules_count mismatch.")
        if L != self.proj_dim:
            raise ValueError(
                f"Checkpoint proj_dim {L} != model {self.proj_dim}"
            )
        if o != self.out_features:
            raise ValueError(
                f"Checkpoint out_features {o} != model {self.out_features}"
            )

        for name in ("foci", "radii", "scatters", "supports", "avg_sq", "R"):
            t = state_dict.get(name)
            if t is None:
                if strict:
                    raise ValueError(f"Strict checkpoint missing {name}.")
                continue
            if not torch.is_tensor(t):
                raise ValueError(f"Checkpoint {name} must be a tensor.")
            if name == "foci":
                if tuple(t.shape) != (r, self.in_features):
                    raise ValueError(f"Checkpoint {name} shape mismatch.")
            elif name == "R":
                if tuple(t.shape) != (r, self.d_ext, self.proj_dim):
                    raise ValueError(f"Checkpoint {name} shape mismatch.")
            else:
                if int(t.shape[0]) != r:
                    raise ValueError(f"Checkpoint {name} length mismatch.")
            if not torch.isfinite(t).all():
                raise ValueError(f"Checkpoint {name} is not finite.")

        if not torch.isfinite(centers).all() or not torch.isfinite(consequent).all():
            raise ValueError("Checkpoint centers/consequent contain non-finite values.")

        if extra is None:
            return

        if int(extra.get("in_features", self.in_features)) != self.in_features:
            raise ValueError("Checkpoint in_features mismatch.")
        if int(extra.get("proj_dim", self.proj_dim)) != self.proj_dim:
            raise ValueError("Checkpoint proj_dim mismatch.")
        if int(extra.get("out_features", self.out_features)) != self.out_features:
            raise ValueError("Checkpoint out_features mismatch.")
        if bool(extra.get("binary", self.binary)) != self.binary:
            raise ValueError("Checkpoint binary flag mismatch.")
        pm = extra.get("projection_mode", self.projection_mode)
        if pm not in _VALID_PROJECTION_MODES:
            raise ValueError(f"Checkpoint projection_mode invalid: {pm!r}")
        if strict and pm != self.projection_mode:
            raise ValueError(
                f"Checkpoint projection_mode {pm!r} != model {self.projection_mode!r}"
            )
        if int(extra.get("rules_count", r)) != r:
            raise ValueError("Checkpoint rules_count mismatch.")

        classes = extra.get("classes")
        if classes is None:
            if strict:
                raise ValueError("Strict checkpoint missing classes.")
        else:
            classes_arr = np.asarray(classes).reshape(-1)
            if self.binary and len(classes_arr) != 2:
                raise ValueError("Binary checkpoint must declare exactly two classes.")
            if (not self.binary) and len(classes_arr) != self.out_features:
                raise ValueError("Checkpoint class count must equal out_features.")

        P = extra.get("P")
        if P is None:
            if strict:
                raise ValueError("Strict checkpoint missing covariance matrices.")
        else:
            if len(P) != r:
                raise ValueError("Checkpoint covariance count mismatch.")
            for i, p in enumerate(P):
                arr = np.asarray(p, dtype=np.float64)
                if arr.shape != (self.proj_dim, self.proj_dim):
                    raise ValueError(f"Checkpoint P[{i}] shape mismatch.")
                if not np.isfinite(arr).all():
                    raise ValueError(f"Checkpoint P[{i}] is not finite.")

        if strict:
            for key in self._STRICT_REQUIRED_EXTRA_KEYS:
                if key not in extra:
                    raise ValueError(f"Strict checkpoint missing {key}.")
            if extra.get("global_mean") is None:
                raise ValueError("Strict checkpoint missing global_mean.")

            # Continuation-sensitive scalars (validated without mutating model).
            try:
                seed_v = int(extra["seed"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Strict checkpoint seed is invalid: {extra['seed']!r}") from exc

            sigma_eps_v = float(extra["sigma_eps"])
            density_eps_v = float(extra["density_epsilon"])
            p_init_v = float(extra["p_init"])
            compression_ratio_v = int(extra["compression_ratio"])
            rules_hint_v = int(extra["rules_hint"])
            n_seen_v = int(extra["n_seen"])
            train_step_v = int(extra["train_step_index"])
            next_rule_v = int(extra["next_rule_index"])
            if not math.isfinite(sigma_eps_v) or sigma_eps_v <= 0.0:
                raise ValueError(f"Strict checkpoint sigma_eps must be finite and > 0, got {sigma_eps_v}")
            if not math.isfinite(density_eps_v) or density_eps_v <= 0.0:
                raise ValueError(
                    f"Strict checkpoint density_epsilon must be finite and > 0, got {density_eps_v}"
                )
            if not math.isfinite(p_init_v) or p_init_v <= 0.0:
                raise ValueError(f"Strict checkpoint p_init must be finite and > 0, got {p_init_v}")
            if compression_ratio_v < 1:
                raise ValueError(
                    f"Strict checkpoint compression_ratio must be >= 1, got {compression_ratio_v}"
                )
            if rules_hint_v < 1:
                raise ValueError(f"Strict checkpoint rules_hint must be >= 1, got {rules_hint_v}")
            mr = extra["max_rules"]
            if mr is not None and int(mr) < 1:
                raise ValueError(f"Strict checkpoint max_rules must be None or >= 1, got {mr}")
            if train_step_v < 0:
                raise ValueError(f"Strict checkpoint train_step_index must be >= 0, got {train_step_v}")
            if next_rule_v < 0:
                raise ValueError(f"Strict checkpoint next_rule_index must be >= 0, got {next_rule_v}")
            if bool(extra.get("is_fitted", False)) and n_seen_v < 1:
                raise ValueError(f"Strict fitted checkpoint requires n_seen >= 1, got {n_seen_v}")
            gas = float(extra["global_avg_sq"])
            if not math.isfinite(gas):
                raise ValueError("Strict checkpoint global_avg_sq is not finite.")
            gm = np.asarray(extra["global_mean"], dtype=np.float64).reshape(-1)
            if gm.shape != (self.in_features,):
                raise ValueError(
                    f"Strict checkpoint global_mean shape {gm.shape} != ({self.in_features},)"
                )
            if not np.isfinite(gm).all():
                raise ValueError("Strict checkpoint global_mean is not finite.")
            flags = list(extra["cloud_updated_flags"])
            if len(flags) != r:
                raise ValueError(
                    f"Strict checkpoint cloud_updated_flags length {len(flags)} != rules_count {r}"
                )

        if pm == PROJECTION_FIXED_PER_CLOUD:
            R = state_dict.get("R")
            if R is None:
                if strict:
                    raise ValueError("Strict checkpoint missing per-cloud R.")
            elif tuple(R.shape) != (r, self.d_ext, self.proj_dim):
                raise ValueError("Checkpoint R shape mismatch.")

        if "inference_R" in extra and extra.get("inference_R") is not None:
            arr = np.asarray(extra["inference_R"], dtype=np.float64)
            if arr.shape != (self.d_ext, self.proj_dim):
                raise ValueError("Checkpoint inference_R shape mismatch.")
            if not np.isfinite(arr).all():
                raise ValueError("Checkpoint inference_R is not finite.")

    def restore_checkpoint(
        self,
        state_dict: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
        *,
        strict: bool = True,
    ):
        """Atomically load tensor + extra state with preflight validation."""
        payload = dict(state_dict)
        if extra is not None:
            payload["_extra_state"] = extra
        return self.load_state_dict(payload, strict=strict)

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """Restore non-tensor state. Strictness comes from ``_checkpoint_load_strict``."""
        strict = self._resolve_checkpoint_strict()
        if not state:
            self.is_fitted_ = False
            return

        if strict:
            for key in self._STRICT_REQUIRED_EXTRA_KEYS:
                if key not in state:
                    raise ValueError(f"Strict extra state missing {key}.")

        if int(state.get("in_features", self.in_features)) != self.in_features:
            raise ValueError("Checkpoint in_features mismatch.")
        if int(state.get("proj_dim", self.proj_dim)) != self.proj_dim:
            raise ValueError("Checkpoint proj_dim mismatch.")
        if int(state.get("out_features", self.out_features)) != self.out_features:
            raise ValueError("Checkpoint out_features mismatch.")
        if bool(state.get("binary", self.binary)) != self.binary:
            raise ValueError("Checkpoint binary flag mismatch.")
        pm = state.get("projection_mode", self.projection_mode)
        if pm not in _VALID_PROJECTION_MODES:
            raise ValueError(f"Checkpoint projection_mode invalid: {pm!r}")
        if strict and pm != self.projection_mode:
            raise ValueError(
                f"Checkpoint projection_mode {pm!r} != model {self.projection_mode!r}"
            )
        self.projection_mode = str(pm)

        # Continuation-sensitive RNG / density parameters.
        missing_cont = [k for k in ("seed", "sigma_eps", "density_epsilon") if k not in state]
        if missing_cont:
            if strict:
                raise ValueError(
                    f"Strict extra state missing continuation keys: {missing_cont}"
                )
            warnings.warn(
                "Legacy VSRP checkpoint missing seed/sigma_eps/density_epsilon "
                f"({missing_cont}); preserving destination constructor values. "
                "Exact online continuation is not guaranteed.",
                RuntimeWarning,
            )
        else:
            try:
                seed_v = int(state["seed"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Checkpoint seed is invalid: {state['seed']!r}") from exc
            sigma_eps_v = float(state["sigma_eps"])
            density_eps_v = float(state["density_epsilon"])
            if not math.isfinite(sigma_eps_v) or sigma_eps_v <= 0.0:
                raise ValueError(
                    f"Checkpoint sigma_eps must be finite and > 0, got {sigma_eps_v}"
                )
            if not math.isfinite(density_eps_v) or density_eps_v <= 0.0:
                raise ValueError(
                    f"Checkpoint density_epsilon must be finite and > 0, got {density_eps_v}"
                )
            self.seed = seed_v
            self.sigma_eps = sigma_eps_v
            self.density_epsilon = density_eps_v

        P = state.get("P")
        if P is None:
            if strict:
                raise ValueError("Strict checkpoint load requires covariance P matrices.")
            warnings.warn(
                "Legacy VSRP checkpoint: missing covariance matrices; leaving empty "
                "and marking model unfitted.",
                RuntimeWarning,
            )
            self._P = []
            self.is_fitted_ = False
            # Still restore whatever else we can, but never claim fitted.
            classes = state.get("classes")
            self._classes = None if classes is None else np.asarray(classes)
            self.rules_count = int(state.get("rules_count", self.rules_count))
            return
        else:
            self._P = [np.asarray(p, dtype=np.float64).copy() for p in P]

        classes = state.get("classes")
        if classes is None and strict:
            raise ValueError("Strict checkpoint missing classes.")
        self._classes = None if classes is None else np.asarray(classes)
        self.rules_count = int(state.get("rules_count", self.rules_count))
        self.rules_hint = int(state.get("rules_hint", self.rules_hint))
        self._n_seen = int(state.get("n_seen", 0))
        gm = state.get("global_mean")
        self._global_mean = None if gm is None else np.asarray(gm, dtype=np.float64).copy()
        self._global_avg_sq = float(state.get("global_avg_sq", 0.0))
        self._next_rule_index = int(
            state.get("next_rule_index", state.get("rules_count", 0))
        )
        self._train_step_index = int(
            state.get("train_step_index", max(self._n_seen - 1, 0))
        )
        if "inference_seed" in state:
            self.inference_seed = int(state["inference_seed"])
        if "inference_R" in state and state["inference_R"] is not None:
            self._inference_R = np.asarray(state["inference_R"], dtype=np.float64).copy()
        else:
            self._inference_R = None
        mr = state.get("max_rules", self.max_rules)
        self.max_rules = None if mr is None else int(mr)
        if "p_init" in state:
            p_init_v = float(state["p_init"])
            if not math.isfinite(p_init_v) or p_init_v <= 0.0:
                raise ValueError(f"Checkpoint p_init must be finite and > 0, got {p_init_v}")
            self.p_init = p_init_v
        if "compression_ratio" in state:
            cr = int(state["compression_ratio"])
            if cr < 1:
                raise ValueError(f"Checkpoint compression_ratio must be >= 1, got {cr}")
            self.compression_ratio = cr
        if "compression_source" in state:
            self.compression_source = str(state["compression_source"])
        flags = state.get("cloud_updated_flags")
        if flags is None:
            if strict and self.rules_count > 0:
                raise ValueError("Strict checkpoint missing cloud_updated_flags.")
            self._cloud_updated_flags = [True] * self.rules_count
            if not strict and self.rules_count > 0:
                warnings.warn(
                    "Legacy VSRP checkpoint: synthesized cloud_updated_flags.",
                    RuntimeWarning,
                )
        else:
            self._cloud_updated_flags = [bool(v) for v in flags]

        if self._n_seen < 0 or self._train_step_index < 0 or self._next_rule_index < 0:
            raise ValueError("Checkpoint counters must be non-negative.")
        if self._global_mean is not None:
            gm = np.asarray(self._global_mean, dtype=np.float64).reshape(-1)
            if gm.shape != (self.in_features,):
                raise ValueError(
                    f"Checkpoint global_mean shape {gm.shape} != ({self.in_features},)"
                )
            self._global_mean = gm
        if not math.isfinite(self._global_avg_sq):
            raise ValueError("Checkpoint global_avg_sq is not finite.")
        if len(self._cloud_updated_flags) != self.rules_count:
            raise ValueError(
                "Checkpoint cloud_updated_flags length does not match rules_count."
            )

        want_fitted = bool(state.get("is_fitted", False))
        if want_fitted and (self._classes is None or not self._P):
            if strict:
                raise ValueError(
                    "Strict checkpoint claims fitted but lacks classes or covariances."
                )
            warnings.warn(
                "Legacy VSRP checkpoint: insufficient metadata to remain fitted.",
                RuntimeWarning,
            )
            self.is_fitted_ = False
        else:
            self.is_fitted_ = want_fitted
        if self.is_fitted_:
            self._validate_live_state(strict=strict)

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = dict(state_dict)
        # Prefer explicit payload; else use PyTorch-packed get_extra_state blob.
        if "_vsrp_extra_state" in state_dict:
            extra = state_dict.pop("_vsrp_extra_state")
            state_dict["_extra_state"] = extra
        else:
            extra = state_dict.get("_extra_state")

        previous_strict = getattr(self, "_checkpoint_load_strict", True)
        self._checkpoint_load_strict = bool(strict)
        # Snapshot destination for failed-load atomicity.
        snap = {
            "rules_count": self.rules_count,
            "is_fitted_": self.is_fitted_,
            "centers": self.centers.detach().clone(),
            "foci": self.foci.detach().clone(),
            "radii": self.radii.detach().clone(),
            "scatters": self.scatters.detach().clone(),
            "supports": self.supports.detach().clone(),
            "avg_sq": self.avg_sq.detach().clone(),
            "R": self.R.detach().clone(),
            "consequent": self.consequent.detach().clone(),
            "P": [p.copy() for p in self._P],
            "classes": None if self._classes is None else np.asarray(self._classes).copy(),
            "n_seen": self._n_seen,
            "global_mean": None
            if self._global_mean is None
            else self._global_mean.copy(),
            "global_avg_sq": self._global_avg_sq,
            "train_step_index": self._train_step_index,
            "next_rule_index": self._next_rule_index,
            "inference_R": None
            if self._inference_R is None
            else self._inference_R.copy(),
            "inference_seed": self.inference_seed,
            "projection_mode": self.projection_mode,
            "cloud_updated_flags": list(self._cloud_updated_flags),
            "proj_dim": self.proj_dim,
            "out_features": self.out_features,
            "p_init": self.p_init,
            "max_rules": self.max_rules,
            "compression_ratio": self.compression_ratio,
            "compression_source": self.compression_source,
            "seed": self.seed,
            "sigma_eps": self.sigma_eps,
            "density_epsilon": self.density_epsilon,
            "rules_hint": self.rules_hint,
            "inference_projection_adaptation": self.inference_projection_adaptation,
            "required_input_scaling": self.required_input_scaling,
            "training_mode": self.training_mode,
            "implementation_status": self.implementation_status,
            "decoder_in": int(self.decoder_linear.in_features),
            "decoder_weight": self.decoder_linear.weight.detach().clone(),
            "decoder_bias": self.decoder_linear.bias.detach().clone(),
        }

        try:
            self._preflight_validate_checkpoint(state_dict, extra, strict=strict)

            kw = self._factory_device_dtype()
            if "foci" not in state_dict and "centers" in state_dict:
                if strict:
                    raise ValueError("Strict checkpoint load: missing foci.")
                state_dict["foci"] = state_dict["centers"].clone()
                warnings.warn(
                    "Legacy VSRP checkpoint: initialized missing foci from centers.",
                    RuntimeWarning,
                )
            if "radii" not in state_dict and "centers" in state_dict:
                if strict:
                    raise ValueError("Strict checkpoint load: missing radii.")
                r = int(state_dict["centers"].shape[0])
                state_dict["radii"] = torch.ones(r, **kw)
                warnings.warn(
                    "Legacy VSRP checkpoint: initialized missing radii to 1.",
                    RuntimeWarning,
                )
            if "scatters" not in state_dict and "centers" in state_dict:
                if strict:
                    raise ValueError("Strict checkpoint load: missing scatters.")
                r = int(state_dict["centers"].shape[0])
                state_dict["scatters"] = torch.full(
                    (r,), math.sqrt(self.density_epsilon), **kw
                )
                warnings.warn(
                    "Legacy VSRP checkpoint: synthesized missing scatters.",
                    RuntimeWarning,
                )
            if "R" not in state_dict and "centers" in state_dict:
                if strict:
                    raise ValueError("Strict checkpoint load: missing R.")
                r = int(state_dict["centers"].shape[0])
                state_dict["R"] = torch.zeros(r, self.d_ext, self.proj_dim, **kw)
                warnings.warn(
                    "Legacy VSRP checkpoint: synthesized zero R buffers.",
                    RuntimeWarning,
                )

            c = state_dict["centers"]
            r = int(c.shape[0])
            self.rules_count = r
            self.register_buffer("centers", torch.zeros(r, self.in_features, **kw))
            self.register_buffer("foci", torch.zeros(r, self.in_features, **kw))
            self.register_buffer("radii", torch.zeros(r, **kw))
            self.register_buffer("scatters", torch.zeros(r, **kw))
            self.register_buffer("supports", torch.zeros(r, **kw))
            self.register_buffer("avg_sq", torch.zeros(r, **kw))
            self.register_buffer(
                "R", torch.zeros(r, self.d_ext, self.proj_dim, **kw)
            )
            self.consequent = nn.Parameter(
                torch.zeros(r, self.proj_dim, self.out_features, **kw),
                requires_grad=False,
            )

            if "decoder_linear.weight" in state_dict:
                w = state_dict["decoder_linear.weight"]
                if torch.is_tensor(w) and w.ndim == 2:
                    in_r = int(w.shape[1])
                    if self.decoder_linear.in_features != in_r:
                        self.decoder_linear = nn.Linear(
                            in_features=in_r,
                            out_features=self.in_features,
                            bias=True,
                            **kw,
                        )
                        for p in self.decoder_linear.parameters():
                            p.requires_grad_(False)

            # Controlled load: buffers already resized to match checkpoint shapes.
            result = super().load_state_dict(state_dict, strict=False)
            # set_extra_state is invoked by Module.load_state_dict via _extra_state.

            missing = [
                k
                for k in result.missing_keys
                if k not in self._COMPAT_MODULE_KEYS and k != "_extra_state"
            ]
            unexpected = [
                k
                for k in result.unexpected_keys
                if k not in self._COMPAT_MODULE_KEYS and k != "_extra_state"
            ]
            if strict and (missing or unexpected):
                raise ValueError(
                    "Strict checkpoint key mismatch after load: "
                    f"missing={missing}, unexpected={unexpected}"
                )

            if extra is None:
                # Tensor-only legacy: never claim fitted; no synthesized covariances.
                self.is_fitted_ = False
                self._P = []
                self._classes = None
                self._cloud_updated_flags = [False] * self.rules_count
            elif strict and self.is_fitted_ and not self._P:
                raise ValueError("Strict checkpoint: fitted model missing P / rules.")
            return result
        except Exception:
            # Restore destination model unchanged on failure.
            self.rules_count = snap["rules_count"]
            self.is_fitted_ = snap["is_fitted_"]
            self.register_buffer("centers", snap["centers"])
            self.register_buffer("foci", snap["foci"])
            self.register_buffer("radii", snap["radii"])
            self.register_buffer("scatters", snap["scatters"])
            self.register_buffer("supports", snap["supports"])
            self.register_buffer("avg_sq", snap["avg_sq"])
            self.register_buffer("R", snap["R"])
            self.consequent = nn.Parameter(snap["consequent"], requires_grad=False)
            self._P = snap["P"]
            self._classes = snap["classes"]
            self._n_seen = snap["n_seen"]
            self._global_mean = snap["global_mean"]
            self._global_avg_sq = snap["global_avg_sq"]
            self._train_step_index = snap["train_step_index"]
            self._next_rule_index = snap["next_rule_index"]
            self._inference_R = snap["inference_R"]
            self.inference_seed = snap["inference_seed"]
            self.projection_mode = snap["projection_mode"]
            self._cloud_updated_flags = snap["cloud_updated_flags"]
            self.proj_dim = snap["proj_dim"]
            self.out_features = snap["out_features"]
            self.p_init = snap["p_init"]
            self.max_rules = snap["max_rules"]
            self.compression_ratio = snap["compression_ratio"]
            self.compression_source = snap["compression_source"]
            self.seed = snap["seed"]
            self.sigma_eps = snap["sigma_eps"]
            self.density_epsilon = snap["density_epsilon"]
            self.rules_hint = snap["rules_hint"]
            self.inference_projection_adaptation = snap[
                "inference_projection_adaptation"
            ]
            self.required_input_scaling = snap["required_input_scaling"]
            self.training_mode = snap["training_mode"]
            self.implementation_status = snap["implementation_status"]
            if self.decoder_linear.in_features != snap["decoder_in"]:
                self.decoder_linear = nn.Linear(
                    in_features=snap["decoder_in"],
                    out_features=self.in_features,
                    bias=True,
                    **self._factory_device_dtype(),
                )
                for p in self.decoder_linear.parameters():
                    p.requires_grad_(False)
            with torch.no_grad():
                self.decoder_linear.weight.copy_(snap["decoder_weight"])
                self.decoder_linear.bias.copy_(snap["decoder_bias"])
            raise
        finally:
            self._checkpoint_load_strict = previous_strict



class SklearnVSRPAnyaEFSWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn facade for VSRPAnyaEFS (custom-fit, no backprop).

    ``predict_proba`` returns softmax of least-squares class scores as
    uncalibrated compatibility scores (not calibrated probabilities).
    """

    def __init__(self, model, device=None, dtype=torch.float32, batch_size: int = 1024):
        self.device = device if device else "cpu"
        self.dtype = dtype
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size
        self.model = model.to(device=self.device, dtype=self.dtype)
        self._sync_fitted_from_model()

    # Known Evaluator / sklearn compatibility kwargs (ignored intentionally).
    _ALLOWED_FIT_KWARGS = frozenset({"device", "random_state", "verbose"})

    def fit(self, X, y, **kwargs):
        unknown = set(kwargs) - self._ALLOWED_FIT_KWARGS
        if unknown:
            raise TypeError(
                f"SklearnVSRPAnyaEFSWrapper.fit() got unexpected keyword "
                f"arguments: {sorted(unknown)}"
            )
        if "device" in kwargs and kwargs["device"] is not None:
            self.device = kwargs["device"]
            self.model = self.model.to(self.device)
        self.model.fit(self._convert_to_tensor(X), self._convert_y(y))
        self._sync_fitted_from_model()
        return self

    def partial_fit(self, X, y, classes=None, **kwargs):
        unknown = set(kwargs) - self._ALLOWED_FIT_KWARGS
        if unknown:
            raise TypeError(
                f"SklearnVSRPAnyaEFSWrapper.partial_fit() got unexpected "
                f"keyword arguments: {sorted(unknown)}"
            )
        if "device" in kwargs and kwargs["device"] is not None:
            self.device = kwargs["device"]
            self.model = self.model.to(self.device)
        self.model.partial_fit(
            self._convert_to_tensor(X), self._convert_y(y), classes=classes
        )
        self._sync_fitted_from_model()
        return self

    def _sync_fitted_from_model(self) -> None:
        if getattr(self.model, "is_fitted_", False) and getattr(
            self.model, "_classes", None
        ) is not None:
            self.classes_ = np.asarray(self.model._classes)
            self.is_fitted_ = True
        else:
            self.classes_ = None
            self.is_fitted_ = False

    def _check_is_fitted(self):
        if not bool(getattr(self, "is_fitted_", False)) or getattr(
            self.model, "_classes", None
        ) is None:
            raise NotFittedError(
                "This SklearnVSRPAnyaEFSWrapper instance is not fitted yet. "
                "Call fit() before predict()."
            )

    def _iter_batches(self, X: Tensor):
        n = X.shape[0]
        bs = max(1, self.batch_size)
        for start in range(0, n, bs):
            yield X[start : start + bs]

    def _forward_logits(self, X) -> Tensor:
        X_t = self._convert_to_tensor(X)
        if X_t.shape[0] == 0:
            return torch.empty((0, self.model.out_features), dtype=self.dtype)
        model_param = next(self.model.parameters())
        model_device = model_param.device
        model_dtype = model_param.dtype
        pin_memory = model_device.type == "cuda"
        was_training = self.model.training
        self.model.eval()
        chunks: List[Tensor] = []
        try:
            with torch.no_grad():
                for xb in self._iter_batches(X_t):
                    xb = xb.to(
                        device=model_device, dtype=model_dtype, non_blocking=pin_memory
                    )
                    # Fixed inference projection is inside model.forward (order-invariant).
                    chunks.append(self.model(xb)[0].detach().cpu())
        finally:
            if was_training:
                self.model.train()
        return torch.cat(chunks, dim=0)

    def predict(self, X):
        self._check_is_fitted()
        logits = self._forward_logits(X)
        classes = np.asarray(self.model._classes)
        if logits.shape[0] == 0:
            return np.empty((0,), dtype=classes.dtype)
        if not torch.isfinite(logits).all():
            raise RuntimeError("VSRP-AnYa-EFS produced non-finite outputs.")
        pred = logits.argmax(dim=1).cpu().numpy()
        return classes[pred]

    def predict_proba(self, X):
        """Uncalibrated softmax compatibility scores over LS class outputs."""
        self._check_is_fitted()
        logits = self._forward_logits(X)
        if logits.shape[0] == 0:
            return np.empty((0, self.model.out_features), dtype=np.float32)
        proba = torch.softmax(logits, dim=1)
        out = proba.detach().cpu().numpy()
        if not np.isfinite(out).all():
            raise RuntimeError("predict_proba produced non-finite values.")
        if out.shape[1] != self.model.out_features:
            raise RuntimeError("predict_proba column count mismatch.")
        if self.model.binary and out.shape[1] != 2:
            raise RuntimeError("Binary predict_proba must return two columns.")
        if not np.allclose(out.sum(axis=1), 1.0, atol=1e-4):
            raise RuntimeError("predict_proba rows must sum to one.")
        return out

    def decision_function(self, X):
        self._check_is_fitted()
        return self._forward_logits(X).detach().cpu().numpy()

    def score(self, X, y):
        self._check_is_fitted()
        return accuracy_score(np.asarray(y).reshape(-1), self.predict(X))

    def _convert_y(self, y):
        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy().reshape(-1)
        if isinstance(y, pd.Series):
            return y.values.reshape(-1)
        return np.asarray(y).reshape(-1)

    def _convert_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            return torch.as_tensor(data, dtype=self.dtype)
        if isinstance(data, torch.Tensor):
            return data.detach().to(device="cpu", dtype=self.dtype)
        if isinstance(data, pd.DataFrame):
            return torch.as_tensor(data.to_numpy(), dtype=self.dtype)
        raise ValueError("Input data must be NumPy, Tensor, or DataFrame.")

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
        self._sync_fitted_from_model()
        return self


# Public aliases used by tests / notebook
_make_rsb_matrix_np = make_rsb_matrix_numpy
