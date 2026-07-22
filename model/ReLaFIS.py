"""ReLaFIS: Relational Linguistic Fuzzy Inference System.

Neuro-fuzzy classifier with relational antecedents and TSK (or Mamdani) consequents.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from torch import nn

__all__ = [
    "ReLaFIS",
    "MamdaniReLaFIS",
    "SklearnReLaFISWrapper",
]


class ReLaFIS(nn.Module):
    """Complete ReLaFIS neuro-fuzzy classifier (relational antecedents + TSK)."""

    N_LINGUISTIC_CATEGORIES = 4

    def __init__(
        self,
        in_features: int,
        rules: int,
        out_features: int,
        binary: bool,
        drop_out_p: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.rules_count = rules
        self.in_features = in_features
        self.out_features = out_features
        self.binary = binary
        self.device = device
        self.dtype = dtype

        if binary:
            self.out_features = out_features = 1

        if drop_out_p != 0.0:
            warnings.warn(
                "drop_out_p is deprecated and ignored; activation dropout "
                "is not applied to rule firing strengths.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.drop_out_p = drop_out_p

        self.mean = nn.Parameter(torch.rand((in_features, rules), **factory_kwargs))
        self.std = nn.Parameter(torch.rand((in_features, rules), **factory_kwargs))
        self.literal = nn.Parameter(
            torch.randn((in_features, rules), **factory_kwargs) * 0.1
        )
        self.temp = nn.Parameter(
            torch.randn((in_features, rules), **factory_kwargs) * 0.1
        )
        self.comb_weight = nn.Parameter(
            torch.randn((in_features, rules), **factory_kwargs) * 0.1
        )

        self.local_slopes = nn.Parameter(
            torch.randn((rules, in_features, out_features), **factory_kwargs) * 0.01
        )
        self.local_biases = nn.Parameter(
            torch.zeros((rules, out_features), **factory_kwargs)
        )

        self.decoder_linear = nn.Linear(
            in_features=rules,
            out_features=in_features,
            bias=True,
            **factory_kwargs,
        )

    @staticmethod
    def _normalized_binary_entropy(weight: torch.Tensor) -> torch.Tensor:
        """Full Bernoulli entropy normalized by log(2), in [0, 1]."""
        eps = torch.finfo(weight.dtype).eps
        safe_weight = weight.clamp(eps, 1.0 - eps)
        entropy = -(
            safe_weight * torch.log(safe_weight)
            + (1.0 - safe_weight) * torch.log(1.0 - safe_weight)
        )
        return entropy / math.log(2.0)

    def _compute_relational_state(self, X: torch.Tensor) -> dict:
        """Centralized antecedent / firing mathematics for ReLaFIS."""
        # X: [batch, features] -> [batch, features, 1]
        X_expanded = X.unsqueeze(-1)

        mean = self.mean.unsqueeze(0)  # [1, features, rules]
        sigma = F.softplus(self.std).clamp_min(1e-3)  # [features, rules]
        sigma_b = sigma.unsqueeze(0)  # [1, features, rules]

        w1 = torch.sigmoid(self.literal)  # [features, rules]
        w2 = torch.sigmoid(self.temp)
        w3 = torch.sigmoid(self.comb_weight)

        w1_b = w1.unsqueeze(0)
        w2_b = w2.unsqueeze(0)
        w3_b = w3.unsqueeze(0)

        r_equal = torch.exp(
            -((X_expanded - mean) ** 2) / (2.0 * sigma_b ** 2)
        )
        r_not_equal = 1.0 - r_equal

        z = (X_expanded - mean) / (math.sqrt(2.0) * sigma_b)
        r_at_least = 0.5 * (1.0 + torch.erf(z))
        r_at_most = 1.0 - r_at_least

        identity_relation = w1_b * r_equal + (1.0 - w1_b) * r_not_equal
        ordering_relation = w2_b * r_at_least + (1.0 - w2_b) * r_at_most
        gamma = w3_b * identity_relation + (1.0 - w3_b) * ordering_relation

        H1 = self._normalized_binary_entropy(w1)
        H2 = self._normalized_binary_entropy(w2)
        rho = w3 * H1 + (1.0 - w3) * H2  # [features, rules]

        gamma_relaxed = rho.unsqueeze(0) + (1.0 - rho.unsqueeze(0)) * gamma

        eps = torch.finfo(X.dtype).eps
        log_firing = torch.log(gamma_relaxed.clamp_min(eps)).sum(dim=1)
        phi = torch.softmax(log_firing, dim=1)

        return {
            "sigma": sigma,
            "w1": w1,
            "w2": w2,
            "w3": w3,
            "H1": H1,
            "H2": H2,
            "rho": rho,
            "r_equal": r_equal,
            "r_not_equal": r_not_equal,
            "r_at_least": r_at_least,
            "r_at_most": r_at_most,
            "gamma": gamma,
            "gamma_relaxed": gamma_relaxed,
            "log_firing": log_firing,
            "phi": phi,
        }

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """Return normalized rule firing strengths φ."""
        return self._compute_relational_state(X)["phi"]

    def tsk(
        self,
        X: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Rule-centered, width-normalized, relaxation-gated TSK consequent."""
        local_coordinates = (
            X.unsqueeze(1) - self.mean.T.unsqueeze(0)
        ) / sigma.T.unsqueeze(0)

        consequent_gate = (1.0 - rho).T
        gated_coordinates = local_coordinates * consequent_gate.unsqueeze(0)

        rule_outputs = torch.einsum(
            "brf,rfo->bro",
            gated_coordinates,
            self.local_slopes,
        )
        rule_outputs = rule_outputs + self.local_biases.unsqueeze(0)

        logits = torch.einsum("br,bro->bo", phi, rule_outputs)
        return logits

    def forward(self, X: torch.Tensor):
        state = self._compute_relational_state(X)
        phi = state["phi"]
        reconstructed_X = self.decoder_linear(phi)
        logits = self.tsk(
            X=X,
            phi=phi,
            rho=state["rho"],
            sigma=state["sigma"],
        )
        return logits, reconstructed_X

    def linguistic_richness(self, per_rule: bool = False):
        """Absolute relational entropy (ARE) over hard-assigned categories.

        Categories (per feature-rule):
            0: equal, 1: not-equal, 2: at-least, 3: at-most
        """
        with torch.no_grad():
            w1 = torch.sigmoid(self.literal)
            w2 = torch.sigmoid(self.temp)
            w3 = torch.sigmoid(self.comb_weight)

            # Hierarchical hard assignment
            identity_branch = w3 >= 0.5
            equal = identity_branch & (w1 >= 0.5)
            not_equal = identity_branch & (w1 < 0.5)
            at_least = (~identity_branch) & (w2 >= 0.5)
            at_most = (~identity_branch) & (w2 < 0.5)

            category = torch.zeros_like(w1, dtype=torch.long)
            category = torch.where(equal, torch.zeros_like(category), category)
            category = torch.where(not_equal, torch.ones_like(category), category)
            category = torch.where(
                at_least, torch.full_like(category, 2), category
            )
            category = torch.where(
                at_most, torch.full_like(category, 3), category
            )

            n_features = self.in_features
            per_rule_H = []
            for rule_idx in range(self.rules_count):
                cats = category[:, rule_idx]
                counts = torch.bincount(cats, minlength=self.N_LINGUISTIC_CATEGORIES)
                probs = counts.float() / float(n_features)
                positive = probs > 0
                H_rule = -(
                    probs[positive] * torch.log(probs[positive])
                ).sum()
                per_rule_H.append(H_rule)

            per_rule_are = torch.stack(per_rule_H)
            mean_are = per_rule_are.mean().item()

        if per_rule:
            return mean_are, per_rule_are
        return mean_are

    def relaxation_rate(self, per_rule: bool = False):
        """Mean semantic relaxation ρ from the shared entropy definition."""
        with torch.no_grad():
            w1 = torch.sigmoid(self.literal)
            w2 = torch.sigmoid(self.temp)
            w3 = torch.sigmoid(self.comb_weight)
            H1 = self._normalized_binary_entropy(w1)
            H2 = self._normalized_binary_entropy(w2)
            rho = w3 * H1 + (1.0 - w3) * H2

            per_rule_rate = rho.mean(dim=0)
            mean_rate = per_rule_rate.mean().item()

        if per_rule:
            return mean_rate, per_rule_rate
        return mean_rate

    def get_interpretable_params(self):
        with torch.no_grad():
            w1 = torch.sigmoid(self.literal)
            w2 = torch.sigmoid(self.temp)
            w3 = torch.sigmoid(self.comb_weight)
            H1 = self._normalized_binary_entropy(w1)
            H2 = self._normalized_binary_entropy(w2)
            rho = w3 * H1 + (1.0 - w3) * H2
            sigma = F.softplus(self.std).clamp_min(1e-3)

            linguistic_richness_mean, linguistic_richness_per_rule = (
                self.linguistic_richness(per_rule=True)
            )
            relaxation_rate_mean, relaxation_rate_per_rule = (
                self.relaxation_rate(per_rule=True)
            )

            def _std(t: torch.Tensor) -> float:
                return t.std(unbiased=False).item()

            stats = {
                "linguistic_richness": linguistic_richness_mean,
                "linguistic_richness_per_rule_std": _std(
                    linguistic_richness_per_rule
                ),
                "relaxation_rate": relaxation_rate_mean,
                "relaxation_rate_per_rule": relaxation_rate_per_rule.cpu().numpy(),
                "relaxation_rate_per_rule_std": _std(relaxation_rate_per_rule),
                "rho_mean": rho.mean().item(),
                "rho_std": _std(rho),
                "rho_min": rho.min().item(),
                "rho_max": rho.max().item(),
                "sigma_mean": sigma.mean().item(),
                "sigma_std": _std(sigma),
                "sigma_min": sigma.min().item(),
                "sigma_max": sigma.max().item(),
                "w1_mean": w1.mean().item(),
                "w1_std": _std(w1),
                "w2_mean": w2.mean().item(),
                "w2_std": _std(w2),
                "w3_mean": w3.mean().item(),
                "w3_std": _std(w3),
                "slope_mean": self.local_slopes.mean().item(),
                "slope_std": _std(self.local_slopes),
                "bias_mean": self.local_biases.mean().item(),
                "bias_std": _std(self.local_biases),
                "center_mean": self.mean.mean().item(),
                "center_std": _std(self.mean),
            }
        return stats


class MamdaniReLaFIS(ReLaFIS):
    """Mamdani-style consequent on shared ReLaFIS antecedents."""

    def __init__(
        self,
        in_features: int,
        rules: int,
        out_features: int,
        binary: bool,
        drop_out_p: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features, rules, out_features, binary, drop_out_p, device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        if binary:
            out_features = 1
        self.mamdani_linear = nn.Linear(
            self.rules_count, out_features, bias=True, **factory_kwargs
        )

    def forward(self, X: torch.Tensor):
        state = self._compute_relational_state(X)
        phi = state["phi"]
        reconstructed_X = self.decoder_linear(phi)
        logits = self.mamdani_linear(phi)
        return logits, reconstructed_X


class SklearnReLaFISWrapper(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper around a trained ReLaFIS model."""

    def __init__(self, model, device=None, dtype=torch.float32):
        self.device = device if device is not None else "cpu"
        self.dtype = dtype
        self.model = model.to(self.device)

    def fit(self, X, y):
        self._check_is_fitted()
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = self._convert_to_tensor(X)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(X)[0]

        if self.model.binary:
            predictions = (torch.sigmoid(logits).squeeze(-1) >= 0.5)
            return predictions.cpu().numpy()

        predictions = torch.softmax(logits, dim=1).argmax(dim=1)
        return predictions.cpu().numpy()

    def predict_proba(self, X):
        self._check_is_fitted()
        X = self._convert_to_tensor(X)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(X)[0]

        if self.model.binary:
            pos = torch.sigmoid(logits)
            if pos.ndim == 1:
                pos = pos.unsqueeze(-1)
            neg = 1.0 - pos
            probs = torch.cat([neg, pos], dim=1)
        else:
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _convert_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=self.dtype, device=self.device)
        elif isinstance(data, torch.Tensor):
            data = data.to(device=self.device, dtype=self.dtype)
        elif isinstance(data, pd.DataFrame):
            data = torch.tensor(
                data.values, dtype=self.dtype, device=self.device
            )
        else:
            raise ValueError(
                "Input data must be a NumPy array, pandas DataFrame, "
                "or a PyTorch tensor."
            )
        return data

    def _check_is_fitted(self):
        if self.model is None:
            raise RuntimeError("Wrapper has no model; call with a trained model.")

    def get_params(self, deep=True):
        return {
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
        }

    def set_params(self, **parameters):
        for key, value in parameters.items():
            setattr(self, key, value)
        if "model" in parameters and self.model is not None:
            self.model = self.model.to(self.device)
        return self
