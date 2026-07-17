"""
VSRP-AnYa-EFS: Very Sparse Random Projection + AnYa Evolving Fuzzy System.

Reference
---------
H. Huang, H.-J. Rong, Z.-X. Yang, C.-M. Vong,
"Jointly evolving and compressing fuzzy system for feature reduction and
classification," Information Sciences, vol. 579, pp. 218–230, 2021.
doi: 10.1016/j.ins.2021.08.003

Paper-aligned reimplementation (not claimed as exact numeric reproduction).

Compression path (paper eqs. 9, 11, 14–18):

  - ``xe = [1, x]``
  - ``h_i = λ_i xe``
  - RSB / VSRP: ``s = √(d+1)``, ``R ∈ {±√s, 0}`` with
    ``P(±√s) = 1/(2s)``, ``P(0) = 1 - 1/s``
  - ``u_i = (1/√L) xe R_i``, ``g_i = λ_i u_i``
  - ``ŷ = Σ_i g_i Q_i``

Training: online AnYa cloud evolution + per-rule FWRLS (no Adam/SGD).
``forward`` returns ``(logits, X.detach())``; reconstruction unused.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from torch import nn

from model.label_utils import (
    append_unseen_classes,
    label_key,
    stable_unique,
    validate_declared_classes,
)


ArrayLike = Union[np.ndarray, torch.Tensor, pd.DataFrame]


def _make_rsb_matrix(
    d_ext: int,
    proj_dim: int,
    generator: torch.Generator | None = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Random sparse-Bernoulli matrix (paper eq. 11), shape (d_ext, L).

    ``s = √(d+1)``; entries ``+√s``, ``0``, ``-√s`` with probabilities
    ``1/(2s)``, ``1 - 1/s``, ``1/(2s)``.
    """
    d_ext = int(d_ext)
    proj_dim = int(proj_dim)
    if d_ext < 1:
        raise ValueError(f"d_ext must be >= 1, got {d_ext}")
    if proj_dim < 1:
        raise ValueError(f"proj_dim must be >= 1, got {proj_dim}")
    if dtype is not None and not (
        dtype == torch.float16
        or dtype == torch.float32
        or dtype == torch.float64
        or (hasattr(dtype, "is_floating_point") and dtype.is_floating_point)
    ):
        # Accept common float dtypes; torch.dtype objects.
        if not str(dtype).startswith("torch.float"):
            raise ValueError(f"RSB dtype must be floating point, got {dtype}")

    s = math.sqrt(d_ext)
    magnitude = math.sqrt(s)
    p = 1.0 / (2.0 * s)
    u = torch.rand(d_ext, proj_dim, generator=generator, device=device, dtype=dtype)
    R = torch.zeros(d_ext, proj_dim, device=device, dtype=dtype)
    R = torch.where(u < p, torch.full_like(R, -magnitude), R)
    R = torch.where(u > 1.0 - p, torch.full_like(R, magnitude), R)
    if not torch.isfinite(R).all():
        raise RuntimeError("RSB matrix contains non-finite values.")
    # Exact ternary alphabet check (within dtype tolerance).
    allowed = torch.tensor(
        [-magnitude, 0.0, magnitude], device=R.device, dtype=R.dtype
    )
    flat = R.reshape(-1)
    ok = torch.any(
        torch.isclose(flat.unsqueeze(1), allowed.unsqueeze(0), rtol=0, atol=1e-5),
        dim=1,
    )
    if not bool(ok.all()):
        raise RuntimeError("RSB matrix contains values outside {-√s, 0, +√s}.")
    return R


class VSRPAnyaEFS(nn.Module):
    """Online VSRP-AnYa-EFS (custom-fit; no Adam / SGD).

    Parameters
    ----------
    in_features, rules, out_features, binary, drop_out_p, device, dtype
        Preserved constructor contract. ``rules`` is only a pre-fit
        placeholder / ``rules_hint`` — it does **not** cap AnYa evolution.
    max_rules
        Optional explicit cloud cap. Default ``None`` (uncapped).
    compression_ratio, proj_dim
        VSRP latent size ``L = ceil((d+1) / compression_ratio)`` unless
        ``proj_dim`` is set explicitly.
    seed
        Base RNG seed; rule ``i`` uses ``seed + rule_creation_index``.
    psi, p_init
        FWRLS forgetting factor and initial covariance scale
        (paper default ``p_init=500``).
    sigma_eps
        Floor for cloud scatter ``Ξ − ‖Γ‖²``.
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
        train_antecedents: bool = True,  # API compat; unused
        device=None,
        dtype=None,
        psi: float = 1.0,
        p_init: float = 500.0,
        sigma_eps: float = 1e-6,
        max_rules: Optional[int] = None,
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
            raise ValueError(
                f"compression_ratio must be >= 1, got {compression_ratio}"
            )
        if not (0.0 < psi <= 1.0):
            raise ValueError(f"psi must be in (0, 1], got {psi}")
        if p_init <= 0.0:
            raise ValueError(f"p_init must be > 0, got {p_init}")
        if sigma_eps <= 0.0:
            raise ValueError(f"sigma_eps must be > 0, got {sigma_eps}")
        if max_rules is not None and int(max_rules) < 1:
            raise ValueError(f"max_rules must be None or >= 1, got {max_rules}")

        self.binary = bool(binary)
        self.in_features = in_features
        if binary:
            self.out_features = 1
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
        self.psi = psi
        self.p_init = p_init
        self.sigma_eps = sigma_eps

        d_ext = in_features + 1
        proj_dim_was_explicit = proj_dim is not None
        if proj_dim is None:
            proj_dim = max(1, math.ceil(d_ext / self.compression_ratio))
        else:
            proj_dim = int(proj_dim)
            if proj_dim < 1:
                raise ValueError(f"proj_dim must be >= 1, got {proj_dim}")
        self.proj_dim = int(proj_dim)
        self.d_ext = d_ext
        self.proj_dim_was_explicit = bool(proj_dim_was_explicit)
        self.compression_source = (
            "explicit_proj_dim" if proj_dim_was_explicit else "compression_ratio"
        )

        # ---- Protocol / Evaluator flags ----
        self.uses_custom_fit = True
        self.supports_backprop_training = False
        self.uses_generic_optimizer = False
        self.uses_reconstruction_loss = False
        self.uses_vsrp_anya_protocol = True
        self.uses_online_anya_evolution = True
        self.uses_local_fwrls = True
        self.required_input_scaling = "minmax_m1_1"
        self.training_mode = "online_anya_vsrp_fwrls"
        self.paper_test_time_updates = False
        self.implementation_status = "paper_aligned_reimplementation"
        self.rsb_distribution = "very_sparse_bernoulli"
        self.rsb_scaling = "1_over_sqrt_L"
        self.is_fitted_ = False

        # Local means Γ_i (membership / local density).
        self.register_buffer(
            "centers", torch.zeros(0, in_features, **factory_kwargs)
        )
        # Focal points ξ_i (evolution / distance) — distinct from Γ.
        self.register_buffer(
            "foci", torch.zeros(0, in_features, **factory_kwargs)
        )
        # Cloud radii ρ_i (eq. 7 recursion); ρ_0 = 1.
        self.register_buffer("radii", torch.zeros(0, **factory_kwargs))
        self.register_buffer("scatters", torch.zeros(0, **factory_kwargs))
        self.register_buffer("supports", torch.zeros(0, **factory_kwargs))
        self.register_buffer("avg_sq", torch.zeros(0, **factory_kwargs))
        self.register_buffer(
            "R", torch.zeros(0, d_ext, self.proj_dim, **factory_kwargs)
        )
        self.consequent = nn.Parameter(
            torch.zeros(0, self.proj_dim, self.out_features, **factory_kwargs),
            requires_grad=False,
        )
        # Inactive compatibility components — not part of VSRP-AnYa-EFS.
        self.decoder_linear = nn.Linear(
            in_features=max(self.rules_hint, 1),
            out_features=in_features,
            bias=True,
            **factory_kwargs,
        )
        for p in self.decoder_linear.parameters():
            p.requires_grad_(False)
        self.drop_out = nn.Dropout(p=drop_out_p)  # never applied

        # Deterministic rule seeding: next rule uses seed + creation_index.
        self._next_rule_index = 0
        self._rule_seed = self.seed  # last used / current counter
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

        self._P: List[np.ndarray] = []
        self._classes: Optional[np.ndarray] = None
        self._n_seen: int = 0
        self._global_mean: Optional[np.ndarray] = None
        self._global_avg_sq: float = 0.0

    @property
    def model_metadata(self) -> Dict[str, Any]:
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
            "psi": self.psi,
            "p_init": self.p_init,
            "rsb_distribution": self.rsb_distribution,
            "rsb_scaling": self.rsb_scaling,
            "test_time_updates": self.paper_test_time_updates,
            "adaptations": [
                "binary one-output {-1,+1} classification interface",
                "sklearn probability-like compatibility scores",
                "train-only preprocessing and hyperparameter selection",
            ],
        }

    @property
    def mean(self) -> torch.Tensor:
        """``(in_features, rules)`` layout of local means Γ."""
        if self.centers.numel() == 0:
            kw = {"device": self.centers.device, "dtype": self.centers.dtype}
            return torch.zeros(self.in_features, 1, **kw)
        return self.centers.transpose(0, 1)

    def local_density(self, X: torch.Tensor) -> torch.Tensor:
        """Local density vs cloud **means** Γ (not foci), shape (batch, rules).

        ``c_i(x) = 1 / (1 + ‖x − Γ_i‖² / σ_i²)``,
        ``σ_i² = max(Ξ_i − ‖Γ_i‖², eps)``.
        """
        if self.rules_count < 1:
            return X.new_zeros(X.shape[0], 0)
        if not torch.isfinite(self.centers).all():
            raise ValueError("Cloud centers are not finite.")
        if not torch.isfinite(self.avg_sq).all():
            raise ValueError("avg_sq is not finite.")
        diff = X.unsqueeze(1) - self.centers.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        center_norm2 = (self.centers ** 2).sum(dim=-1)
        sigma2 = (self.avg_sq - center_norm2).clamp(min=self.sigma_eps)
        if (sigma2 <= 0).any() or not torch.isfinite(sigma2).all():
            raise ValueError("Local density sigma^2 must be finite and positive.")
        c = 1.0 / (1.0 + dist2 / sigma2.unsqueeze(0))
        if not torch.isfinite(c).all():
            raise ValueError("Local densities are not finite.")
        if (c <= 0).any() or (c > 1.0 + 1e-5).any():
            raise ValueError("Local densities must lie in (0, 1].")
        return c

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """Normalised local densities λ_i (paper eq. 13)."""
        self._require_fitted()
        c = self.local_density(X)
        if c.shape[1] == 0:
            raise RuntimeError("Cannot encode with zero rules.")
        firing = F.normalize(c, p=1, dim=1)
        if not torch.isfinite(firing).all():
            raise ValueError("Normalized firing is not finite.")
        row = firing.sum(dim=1)
        if not torch.allclose(row, torch.ones_like(row), atol=1e-4):
            raise RuntimeError("Normalized firing rows must sum to one.")
        return firing

    def compressed_antecedents(
        self, X: torch.Tensor, firing: torch.Tensor
    ) -> torch.Tensor:
        """``g_i = λ_i * u_i`` with ``u_i = (1/√L) xe R_i``, ``xe = [1, x]``."""
        B = X.shape[0]
        ones = torch.ones(B, 1, device=X.device, dtype=X.dtype)
        xe = torch.cat([ones, X], dim=1)
        u = torch.einsum("bd,rdl->brl", xe, self.R) / math.sqrt(self.proj_dim)
        return firing.unsqueeze(-1) * u

    def uncompressed_projection(self, X: torch.Tensor) -> torch.Tensor:
        """``u_i = (1/√L) xe R_i`` (FWRLS regressor without λ)."""
        B = X.shape[0]
        ones = torch.ones(B, 1, device=X.device, dtype=X.dtype)
        xe = torch.cat([ones, X], dim=1)
        return torch.einsum("bd,rdl->brl", xe, self.R) / math.sqrt(self.proj_dim)

    def consequent_output(self, g: torch.Tensor) -> torch.Tensor:
        """``Σ_i g_i Q_i``."""
        return torch.einsum("brl,rlo->bo", g, self.consequent)

    def linguistic_richness(self, per_rule: bool = False):
        r = max(self.rules_count, 1)
        with torch.no_grad():
            entropies = torch.full(
                (r,), float("nan"), dtype=torch.float64, device=self.centers.device
            )
        if per_rule:
            return float("nan"), entropies
        return float("nan")

    def forward(self, X: torch.Tensor):
        self._require_fitted()
        firing = self.encode(X)
        g = self.compressed_antecedents(X, firing)
        y = self.consequent_output(g)
        return y, X.detach()

    def fit(self, X: ArrayLike, y: ArrayLike):
        X_np, y_np = self._to_numpy_xy(X, y)
        if X_np.shape[0] < 1:
            raise ValueError("Cannot fit on an empty training set.")
        classes = stable_unique(y_np)
        if self.binary:
            if len(classes) != 2:
                raise ValueError(
                    "Binary VSRP-AnYa-EFS requires exactly two classes."
                )
        else:
            if len(classes) != self.out_features:
                raise ValueError(
                    f"VSRP-AnYa-EFS was constructed with {self.out_features} "
                    f"outputs but training contains {len(classes)} classes."
                )
        self._classes = classes
        self._reset_state()
        # Preserve validated schema after reset.
        self._classes = classes

        for i in range(X_np.shape[0]):
            self._online_step(X_np[i], y_np[i])

        self._sync_torch_state()
        self.is_fitted_ = self.rules_count >= 1
        if not self.is_fitted_:
            raise RuntimeError("Fitting produced zero rules.")
        return self

    def partial_fit(
        self, X: ArrayLike, y: ArrayLike, classes: Optional[ArrayLike] = None
    ):
        X_np, y_np = self._to_numpy_xy(X, y)
        self._initialize_or_validate_classes(classes, y_np)

        if not self.is_fitted_ or self.rules_count < 1:
            # First-call path: may use declared schema with subset labels.
            declared = self._classes
            self._reset_state()
            self._classes = declared
            if self.binary and (
                self._classes is None or len(self._classes) != 2
            ):
                raise ValueError(
                    "Binary VSRP-AnYa-EFS requires exactly two classes."
                )
            if not self.binary:
                if self._classes is None:
                    raise ValueError("Multiclass partial_fit has no class schema.")
                if len(self._classes) != self.out_features:
                    raise ValueError(
                        f"Declared {len(self._classes)} classes but "
                        f"out_features={self.out_features}."
                    )
            for i in range(X_np.shape[0]):
                self._online_step(X_np[i], y_np[i])
            self._sync_torch_state()
            self.is_fitted_ = self.rules_count >= 1
            return self

        for i in range(X_np.shape[0]):
            self._online_step(X_np[i], y_np[i])

        self._sync_torch_state()
        self.is_fitted_ = self.rules_count >= 1
        return self

    def _initialize_or_validate_classes(
        self, classes: Optional[ArrayLike], y_np: np.ndarray
    ) -> None:
        if classes is not None:
            declared = validate_declared_classes(classes)
            if self.binary and len(declared) != 2:
                raise ValueError(
                    "Binary VSRP-AnYa-EFS classes= must declare exactly two classes."
                )
            if self._classes is None:
                self._classes = declared
                if not self.binary:
                    if len(declared) != self.out_features:
                        raise ValueError(
                            f"Declared {len(declared)} classes but "
                            f"out_features={self.out_features}."
                        )
            else:
                existing = [label_key(c) for c in np.asarray(self._classes)]
                declared_keys = [label_key(c) for c in declared]
                known = [k for k in declared_keys if k in set(existing)]
                if known != [k for k in existing if k in set(declared_keys)]:
                    raise ValueError(
                        "classes= would reorder existing VSRP class columns; "
                        f"existing={existing!r}, declared={declared_keys!r}"
                    )
                self._classes = append_unseen_classes(self._classes, declared)
            if self.binary and len(self._classes) > 2:
                raise ValueError(
                    "Binary VSRP-AnYa-EFS cannot accept more than two classes."
                )
            return

        if self._classes is None:
            self._classes = stable_unique(y_np)
        else:
            self._classes = append_unseen_classes(self._classes, y_np)
        if self.binary and len(self._classes) > 2:
            raise ValueError(
                "Binary VSRP-AnYa-EFS cannot accept more than two distinct classes."
            )

    def _require_fitted(self) -> None:
        if not self.is_fitted_ or self.rules_count < 1:
            raise RuntimeError(
                "VSRPAnyaEFS is not fitted. Call fit(X, y) before forward/encode."
            )

    def _factory_device_dtype(self) -> Dict[str, Any]:
        return {
            "device": self.centers.device,
            "dtype": self.centers.dtype
            if self.centers.dtype is not None
            else torch.float32,
        }

    def _to_numpy_xy(
        self, X: ArrayLike, y: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        """Reset evolving state while preserving validated out_features / schema."""
        self.rules_count = 0
        self._P = []
        self._n_seen = 0
        self._global_mean = None
        self._global_avg_sq = 0.0
        self.is_fitted_ = False
        self._next_rule_index = 0
        self._rule_seed = self.seed
        kw = self._factory_device_dtype()
        self.register_buffer("centers", torch.zeros(0, self.in_features, **kw))
        self.register_buffer("foci", torch.zeros(0, self.in_features, **kw))
        self.register_buffer("radii", torch.zeros(0, **kw))
        self.register_buffer("scatters", torch.zeros(0, **kw))
        self.register_buffer("supports", torch.zeros(0, **kw))
        self.register_buffer("avg_sq", torch.zeros(0, **kw))
        self.register_buffer(
            "R", torch.zeros(0, self.d_ext, self.proj_dim, **kw)
        )
        self.consequent = nn.Parameter(
            torch.zeros(0, self.proj_dim, self.out_features, **kw),
            requires_grad=False,
        )
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

    def _target_vector(self, y_i) -> np.ndarray:
        if self.binary:
            if self._classes is None or len(self._classes) != 2:
                raise ValueError(
                    "Binary VSRP-AnYa-EFS requires exactly two stored classes."
                )
            key = label_key(y_i)
            negative = label_key(self._classes[0])
            positive = label_key(self._classes[1])
            if key == negative:
                return np.asarray([-1.0], dtype=np.float64)
            if key == positive:
                return np.asarray([1.0], dtype=np.float64)
            raise ValueError(f"Unknown binary class label: {key!r}")

        if self._classes is None:
            raise ValueError("Multiclass VSRP-AnYa-EFS has no stored class mapping.")
        if len(self._classes) != self.out_features:
            raise ValueError(
                f"Class count {len(self._classes)} != out_features {self.out_features}."
            )
        class_to_index = {
            label_key(label): index
            for index, label in enumerate(np.asarray(self._classes).reshape(-1))
        }
        key = label_key(y_i)
        if key not in class_to_index:
            raise ValueError(f"Unknown multiclass label: {key!r}")
        t = np.zeros(self.out_features, dtype=np.float64)
        t[class_to_index[key]] = 1.0
        return t

    def _global_density(self, x: np.ndarray) -> float:
        assert self._global_mean is not None
        diff = x - self._global_mean
        dist2 = float(np.dot(diff, diff))
        center_n2 = float(np.dot(self._global_mean, self._global_mean))
        sigma2 = max(self._global_avg_sq - center_n2, self.sigma_eps)
        return 1.0 / (1.0 + dist2 / sigma2)

    def _local_density_np(self, x: np.ndarray, i: int) -> float:
        c = self.centers[i].detach().cpu().numpy()
        diff = x - c
        dist2 = float(np.dot(diff, diff))
        avg_sq = float(self.avg_sq[i].detach().cpu().item())
        center_n2 = float(np.dot(c, c))
        sigma2 = max(avg_sq - center_n2, self.sigma_eps)
        return 1.0 / (1.0 + dist2 / sigma2)

    def _varrho(self, i: int) -> float:
        """Local scatter ϱ_i = √(Ξ_i − ‖Γ_i‖²)."""
        g = self.centers[i].detach().cpu().numpy()
        avg_sq = float(self.avg_sq[i].detach().cpu().item())
        return math.sqrt(max(avg_sq - float(np.dot(g, g)), self.sigma_eps))

    def _update_radius_eq7(self, i: int) -> None:
        """Radius recursion: ρ ← ½(ρ_prev + ϱ), ρ_0 = 1.

        Applied exactly once after a cloud is created or updated with a sample
        (paper eq. 7). Not re-applied in ``_sync_torch_state``.
        """
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

    def _new_rsb(self) -> torch.Tensor:
        """Independent RSB for the next rule; seed = base + creation_index."""
        creation_index = int(self._next_rule_index)
        self._next_rule_index += 1
        rule_seed = int(self.seed) + creation_index
        self._rule_seed = rule_seed
        gen = torch.Generator(device="cpu")
        gen.manual_seed(rule_seed)
        kw = self._factory_device_dtype()
        R = _make_rsb_matrix(
            self.d_ext,
            self.proj_dim,
            generator=gen,
            device="cpu",
            dtype=torch.float32,
        )
        return R.to(**kw)

    def _add_cloud(self, x: np.ndarray) -> None:
        kw = self._factory_device_dtype()
        x_t = torch.tensor(x, **kw).unsqueeze(0)
        x_n2 = float(np.dot(x, x))
        R_new = self._new_rsb().unsqueeze(0)
        Q_new = torch.zeros(1, self.proj_dim, self.out_features, **kw)
        rho0 = torch.ones(1, **kw)
        v0 = torch.full((1,), math.sqrt(self.sigma_eps), **kw)

        if self.rules_count == 0:
            self.register_buffer("centers", x_t.clone())
            self.register_buffer("foci", x_t.clone())
            self.register_buffer("radii", rho0.clone())
            self.register_buffer("scatters", v0.clone())
            self.register_buffer("supports", torch.ones(1, **kw))
            self.register_buffer("avg_sq", torch.tensor([x_n2], **kw))
            self.register_buffer("R", R_new)
            self.consequent = nn.Parameter(Q_new, requires_grad=False)
        else:
            self.register_buffer(
                "centers", torch.cat([self.centers, x_t], dim=0)
            )
            self.register_buffer("foci", torch.cat([self.foci, x_t], dim=0))
            self.register_buffer(
                "radii", torch.cat([self.radii, rho0], dim=0)
            )
            self.register_buffer(
                "scatters", torch.cat([self.scatters, v0], dim=0)
            )
            self.register_buffer(
                "supports",
                torch.cat([self.supports, torch.ones(1, **kw)], dim=0),
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

        self._P.append(self.p_init * np.eye(self.proj_dim, dtype=np.float64))
        self.rules_count = int(self.centers.shape[0])
        self._update_radius_eq7(self.rules_count - 1)

    def _update_cloud(self, i: int, x: np.ndarray) -> None:
        """Update mean Γ / Ξ; replace focal ξ of nearest cloud with x."""
        m = float(self.supports[i].detach().cpu().item())
        c = self.centers[i].detach().cpu().numpy()
        avg_sq = float(self.avg_sq[i].detach().cpu().item())
        m_new = m + 1.0
        c_new = (m / m_new) * c + x / m_new
        x_n2 = float(np.dot(x, x))
        avg_sq_new = (m / m_new) * avg_sq + x_n2 / m_new

        kw = self._factory_device_dtype()
        with torch.no_grad():
            self.centers[i] = torch.tensor(c_new, **kw)
            self.supports[i] = m_new
            self.avg_sq[i] = avg_sq_new
            self.foci[i] = torch.tensor(x, **kw)
        self._update_radius_eq7(i)

    def _should_add_cloud(self, x: np.ndarray) -> bool:
        if self.rules_count < 1:
            return True
        if self.max_rules is not None and self.rules_count >= self.max_rules:
            return False

        dens_x = self._global_density(x)
        # Cloud global density uses local means Γ_i, not foci ξ_i.
        means = self.centers.detach().cpu().numpy()
        dens_means = np.array(
            [self._global_density(means[i]) for i in range(self.rules_count)]
        )
        denser_than_all = bool(np.all(dens_x > dens_means + 1e-15))
        less_than_all = bool(np.all(dens_x < dens_means - 1e-15))
        density_extreme = denser_than_all or less_than_all

        # Distance criterion uses foci ξ_i and radii ρ_i.
        foci = self.foci.detach().cpu().numpy()
        dists = np.linalg.norm(foci - x.reshape(1, -1), axis=1)
        radii = self.radii.detach().cpu().numpy()
        far_from_all = bool(np.all(dists > radii + 1e-15))

        # OR of the two addition criteria.
        return density_extreme or far_from_all

    def _firing_np(self, x: np.ndarray) -> np.ndarray:
        if self.rules_count < 1:
            return np.zeros(0, dtype=np.float64)
        c = np.array(
            [self._local_density_np(x, i) for i in range(self.rules_count)],
            dtype=np.float64,
        )
        if not np.isfinite(c).all() or np.all(c <= 0):
            raise RuntimeError(
                "Local densities became numerically unusable."
            )
        s = c.sum()
        if s <= 0:
            raise RuntimeError("Local density sum is non-positive.")
        return c / s

    def _project_u(self, x: np.ndarray) -> np.ndarray:
        xe = np.concatenate([[1.0], x]).astype(np.float64)
        R = self.R.detach().cpu().numpy().astype(np.float64)
        scale = 1.0 / math.sqrt(self.proj_dim)
        return scale * np.einsum("rdl,d->rl", R, xe)

    def _fwrls_update(self, lam: np.ndarray, U: np.ndarray, target: np.ndarray) -> None:
        """Per-rule FWRLS with local residual ``e_i = y − u_i Q_i``."""
        Q = self.consequent.detach().cpu().numpy().astype(np.float64)
        psi = max(self.psi, 1e-12)
        L = self.proj_dim
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if not np.isfinite(target).all():
            raise ValueError("FWRLS target is not finite.")
        if len(self._P) != self.rules_count:
            raise RuntimeError("P matrix count does not match rules_count.")

        for i in range(self.rules_count):
            w = float(lam[i])
            if not math.isfinite(w) or w < 0:
                raise ValueError(f"lambda[{i}] must be finite and >= 0, got {w}")
            weight = max(w, 1e-12)
            u = U[i].reshape(L, 1)
            if not np.isfinite(u).all():
                raise ValueError(f"u[{i}] is not finite.")
            P = self._P[i]
            if P.shape != (L, L):
                raise ValueError(f"P[{i}] shape {P.shape} != {(L, L)}")
            if Q[i].shape != (L, self.out_features):
                raise ValueError(
                    f"Q[{i}] shape {Q[i].shape} != {(L, self.out_features)}"
                )

            pred_i = (U[i] @ Q[i]).reshape(-1)
            err_i = (target - pred_i).reshape(1, -1)

            Pu = P @ u
            denom = (psi / weight) + float(np.asarray(u.T @ Pu).reshape(-1)[0])
            if not math.isfinite(denom) or denom <= 0:
                raise RuntimeError(f"FWRLS denominator invalid: {denom}")
            K = Pu / denom
            if not np.isfinite(K).all():
                raise RuntimeError("FWRLS Kalman gain is not finite.")
            Q[i] = Q[i] + K @ err_i
            P_new = (P - (K @ u.T) @ P) / psi
            P_new = 0.5 * (P_new + P_new.T)
            if not np.isfinite(P_new).all() or not np.isfinite(Q[i]).all():
                raise RuntimeError("FWRLS produced non-finite P or Q.")
            self._P[i] = P_new

        kw = self._factory_device_dtype()
        with torch.no_grad():
            self.consequent.data.copy_(torch.tensor(Q, **kw))

    def _online_step(self, x: np.ndarray, y_i) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.in_features:
            raise ValueError(
                f"Sample has {x.shape[0]} features, expected {self.in_features}"
            )
        if not np.isfinite(x).all():
            raise ValueError("Sample contains NaN or infinite values.")

        # Append unseen class before creating the target.
        if self._classes is None:
            self._classes = stable_unique(np.asarray([y_i]))
        else:
            prev_n = len(self._classes)
            self._classes = append_unseen_classes(self._classes, np.asarray([y_i]))
            if self.binary and len(self._classes) > 2:
                raise ValueError(
                    "Binary VSRP-AnYa-EFS cannot accept more than two distinct classes."
                )
            if (not self.binary) and len(self._classes) > prev_n:
                self._expand_outputs(len(self._classes))

        target = self._target_vector(y_i)
        self._update_global(x)

        if self.rules_count < 1:
            self._add_cloud(x)
            lam = np.ones(1, dtype=np.float64)
            U = self._project_u(x)
            self._fwrls_update(lam, U, target)
            return

        if self._should_add_cloud(x):
            self._add_cloud(x)
        else:
            foci = self.foci.detach().cpu().numpy()
            j = int(np.argmin(np.linalg.norm(foci - x.reshape(1, -1), axis=1)))
            self._update_cloud(j, x)

        lam = self._firing_np(x)
        U = self._project_u(x)
        self._fwrls_update(lam, U, target)

    def _expand_outputs(self, n_classes: int) -> None:
        if self.binary:
            return
        if n_classes <= self.out_features:
            return
        add = n_classes - self.out_features
        self.out_features = n_classes
        if self.rules_count < 1:
            return
        kw = self._factory_device_dtype()
        pad = torch.zeros(self.rules_count, self.proj_dim, add, **kw)
        self.consequent = nn.Parameter(
            torch.cat([self.consequent.data, pad], dim=2),
            requires_grad=False,
        )

    def _sync_torch_state(self) -> None:
        """No-op: radii / scatters are updated online in add/update paths.

        Re-applying Eq. (7) here would shrink ``ρ ← ½(ρ + ϱ)`` with no new
        sample assignment and corrupt the evolved radii.
        """
        return

    def snapshot_state(self) -> Dict[str, Any]:
        """Hashable-ish numeric snapshot for test-time freeze checks."""
        return {
            "rules_count": int(self.rules_count),
            "centers": self.centers.detach().cpu().numpy().copy(),
            "foci": self.foci.detach().cpu().numpy().copy(),
            "radii": self.radii.detach().cpu().numpy().copy(),
            "supports": self.supports.detach().cpu().numpy().copy(),
            "avg_sq": self.avg_sq.detach().cpu().numpy().copy(),
            "R": self.R.detach().cpu().numpy().copy(),
            "Q": self.consequent.detach().cpu().numpy().copy(),
            "P": [p.copy() for p in self._P],
            "n_seen": int(self._n_seen),
            "global_mean": None
            if self._global_mean is None
            else self._global_mean.copy(),
            "global_avg_sq": float(self._global_avg_sq),
            "classes": None
            if self._classes is None
            else np.asarray(self._classes).copy(),
            "next_rule_index": int(self._next_rule_index),
            "rule_seed": int(self._rule_seed),
        }

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "P": [p.copy() for p in self._P],
            "classes": None
            if self._classes is None
            else np.asarray(self._classes).tolist(),
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
            "next_rule_index": int(self._next_rule_index),
            "rule_seed": int(self._rule_seed),
            "psi": float(self.psi),
            "p_init": float(self.p_init),
            "sigma_eps": float(self.sigma_eps),
            "max_rules": self.max_rules,
            "n_seen": int(self._n_seen),
            "global_mean": None
            if self._global_mean is None
            else self._global_mean.copy(),
            "global_avg_sq": float(self._global_avg_sq),
            "compression_source": self.compression_source,
            "implementation_status": self.implementation_status,
            "training_mode": self.training_mode,
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        if not state:
            self.is_fitted_ = False
            return
        try:
            if int(state.get("in_features", self.in_features)) != self.in_features:
                raise ValueError("Checkpoint in_features mismatch.")
            if int(state.get("proj_dim", self.proj_dim)) != self.proj_dim:
                raise ValueError("Checkpoint proj_dim mismatch.")
            if bool(state.get("binary", self.binary)) != self.binary:
                raise ValueError("Checkpoint binary flag mismatch.")

            P = state.get("P")
            self._P = (
                []
                if P is None
                else [np.asarray(p, dtype=np.float64).copy() for p in P]
            )
            classes = state.get("classes")
            self._classes = None if classes is None else np.asarray(classes)
            self.rules_count = int(state.get("rules_count", self.rules_count))
            self.rules_hint = int(state.get("rules_hint", self.rules_hint))
            self.out_features = int(state.get("out_features", self.out_features))
            self._n_seen = int(state.get("n_seen", 0))
            gm = state.get("global_mean")
            self._global_mean = (
                None if gm is None else np.asarray(gm, dtype=np.float64).copy()
            )
            self._global_avg_sq = float(state.get("global_avg_sq", 0.0))
            self._next_rule_index = int(
                state.get("next_rule_index", state.get("rules_count", 0))
            )
            self._rule_seed = int(state.get("rule_seed", self.seed))
            mr = state.get("max_rules", self.max_rules)
            self.max_rules = None if mr is None else int(mr)
            if "p_init" in state:
                self.p_init = float(state["p_init"])
            if "psi" in state:
                self.psi = float(state["psi"])
            if "sigma_eps" in state:
                self.sigma_eps = float(state["sigma_eps"])

            complete = bool(state.get("is_fitted", False))
            if complete:
                self._validate_checkpoint_tensors(strict=True)
            self.is_fitted_ = complete
        except Exception as exc:
            self.is_fitted_ = False
            warnings.warn(
                f"VSRP checkpoint extra state incomplete/inconsistent: {exc}",
                RuntimeWarning,
            )
            raise

    def _validate_checkpoint_tensors(self, *, strict: bool) -> None:
        r = int(self.rules_count)
        if r < 1:
            raise ValueError("Fitted checkpoint must have rules_count >= 1.")
        if tuple(self.centers.shape) != (r, self.in_features):
            raise ValueError("centers shape mismatch.")
        if tuple(self.foci.shape) != (r, self.in_features):
            raise ValueError("foci shape mismatch.")
        if int(self.radii.shape[0]) != r or int(self.supports.shape[0]) != r:
            raise ValueError("radii/supports length mismatch.")
        if tuple(self.R.shape) != (r, self.d_ext, self.proj_dim):
            raise ValueError("R shape mismatch.")
        if tuple(self.consequent.shape) != (r, self.proj_dim, self.out_features):
            raise ValueError("consequent shape mismatch.")
        if len(self._P) != r:
            raise ValueError("P count mismatch.")
        for i, P in enumerate(self._P):
            if P.shape != (self.proj_dim, self.proj_dim):
                raise ValueError(f"P[{i}] shape mismatch.")
            if not np.isfinite(P).all():
                raise ValueError(f"P[{i}] is not finite.")
            if not np.allclose(P, P.T, atol=1e-5):
                raise ValueError(f"P[{i}] is not symmetric.")
        if self._classes is None:
            raise ValueError("Fitted checkpoint missing classes.")
        if self.binary and len(self._classes) != 2:
            raise ValueError("Binary fitted model must have exactly two classes.")
        if (not self.binary) and len(self._classes) != self.out_features:
            raise ValueError("Multiclass class count must equal out_features.")
        for name, t in (
            ("centers", self.centers),
            ("foci", self.foci),
            ("radii", self.radii),
            ("supports", self.supports),
            ("avg_sq", self.avg_sq),
            ("R", self.R),
            ("consequent", self.consequent),
        ):
            if not torch.isfinite(t).all():
                raise ValueError(f"{name} contains non-finite values.")
        if (self.supports <= 0).any():
            raise ValueError("supports must be positive.")
        if (self.radii < 0).any():
            raise ValueError("radii must be non-negative.")

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = dict(state_dict)
        kw = self._factory_device_dtype()

        if "centers" in state_dict and torch.is_tensor(state_dict["centers"]):
            c = state_dict["centers"]
            if c.ndim == 2:
                r = int(c.shape[0])
                self.rules_count = r
                self.register_buffer(
                    "centers", torch.zeros(r, self.in_features, **kw)
                )
                self.register_buffer(
                    "foci", torch.zeros(r, self.in_features, **kw)
                )
                self.register_buffer("radii", torch.zeros(r, **kw))
                self.register_buffer("scatters", torch.zeros(r, **kw))
                self.register_buffer("supports", torch.zeros(r, **kw))
                self.register_buffer("avg_sq", torch.zeros(r, **kw))
                self.register_buffer(
                    "R", torch.zeros(r, self.d_ext, self.proj_dim, **kw)
                )

        if "foci" not in state_dict and "centers" in state_dict:
            state_dict["foci"] = state_dict["centers"].clone()
            if not strict:
                warnings.warn(
                    "Legacy VSRP checkpoint: initialized missing foci from centers.",
                    RuntimeWarning,
                )
        if "radii" not in state_dict and "centers" in state_dict:
            r = int(state_dict["centers"].shape[0])
            state_dict["radii"] = torch.ones(r, **kw)
            if not strict:
                warnings.warn(
                    "Legacy VSRP checkpoint: initialized missing radii to 1.",
                    RuntimeWarning,
                )

        if "consequent" in state_dict and torch.is_tensor(state_dict["consequent"]):
            q = state_dict["consequent"]
            if q.ndim == 3:
                r, L, o = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
                self.rules_count = r
                self.proj_dim = L
                self.out_features = o
                self.consequent = nn.Parameter(
                    torch.zeros(r, L, o, **kw), requires_grad=False
                )
                if self.R.shape[-1] != L or self.R.shape[0] != r:
                    self.register_buffer(
                        "R", torch.zeros(r, self.d_ext, L, **kw)
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

        missing = super().load_state_dict(state_dict, strict=strict)
        if self.rules_count < 1 or (not self._P and self.is_fitted_):
            self.is_fitted_ = False
        return missing


class SklearnVSRPAnyaEFSWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn facade for VSRPAnyaEFS (custom-fit, no backprop).

    ``predict_proba`` uses sigmoid/softmax of least-squares outputs as
    uncalibrated compatibility scores.
    """

    def __init__(self, model, device=None, dtype=torch.float32):
        self.device = device if device else "cpu"
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=self.dtype)
        if getattr(self.model, "is_fitted_", False) and getattr(
            self.model, "_classes", None
        ) is not None:
            self.classes_ = np.asarray(self.model._classes)
            self.is_fitted_ = True
        else:
            self.classes_ = None
            self.is_fitted_ = False

    def fit(self, X, y):
        self.model.fit(self._convert_to_tensor(X), self._convert_y(y))
        self.classes_ = np.asarray(self.model._classes)
        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y, classes=None):
        self.model.partial_fit(
            self._convert_to_tensor(X), self._convert_y(y), classes=classes
        )
        self.classes_ = (
            None
            if self.model._classes is None
            else np.asarray(self.model._classes)
        )
        self.is_fitted_ = bool(self.model.is_fitted_)
        return self

    def _check_is_fitted(self):
        if not getattr(self.model, "is_fitted_", False):
            raise NotFittedError("VSRP-AnYa-EFS has not been fitted.")
        classes = getattr(self.model, "_classes", None)
        if classes is None or len(classes) == 0:
            raise NotFittedError("VSRP-AnYa-EFS has no stored class mapping.")

    def predict(self, X):
        self._check_is_fitted()
        X_t = self._convert_to_tensor(X)
        with torch.no_grad():
            raw = self.model(X_t)[0]
        if not torch.isfinite(raw).all():
            raise RuntimeError("VSRP-AnYa-EFS produced non-finite outputs.")
        classes = np.asarray(self.model._classes)
        if self.model.binary:
            raw_np = raw.detach().cpu().numpy().reshape(-1)
            return np.where(raw_np > 0.0, classes[1], classes[0])
        idx = raw.argmax(dim=1).detach().cpu().numpy()
        return classes[idx]

    def predict_proba(self, X):
        """Uncalibrated compatibility scores (sigmoid / softmax)."""
        self._check_is_fitted()
        X_t = self._convert_to_tensor(X)
        with torch.no_grad():
            raw = self.model(X_t)[0]
            if self.model.binary:
                pos = torch.sigmoid(raw.reshape(-1, 1))
                proba = torch.cat([1.0 - pos, pos], dim=1)
            else:
                proba = torch.softmax(raw, dim=1)
        out = proba.detach().cpu().numpy()
        if not np.isfinite(out).all():
            raise RuntimeError("predict_proba produced non-finite values.")
        if not np.allclose(out.sum(axis=1), 1.0, atol=1e-4):
            raise RuntimeError("predict_proba rows must sum to one.")
        return out

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
            return torch.tensor(data, dtype=self.dtype, device=self.device)
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=self.dtype)
        if isinstance(data, pd.DataFrame):
            return torch.tensor(data.values, dtype=self.dtype, device=self.device)
        raise ValueError("Input data must be NumPy, Tensor, or DataFrame.")

    def get_params(self, deep=True):
        return {
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
        }

    def set_params(self, **parameters):
        unknown = set(parameters) - {"model", "device", "dtype"}
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        if "device" in parameters:
            self.device = parameters["device"]
        if "dtype" in parameters:
            self.dtype = parameters["dtype"]
        if "model" in parameters:
            self.model = parameters["model"]
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        if getattr(self.model, "is_fitted_", False) and self.model._classes is not None:
            self.classes_ = np.asarray(self.model._classes)
            self.is_fitted_ = True
        return self
