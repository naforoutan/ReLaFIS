import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd


class ReLaFIS(nn.Module):

    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, drop_out_p=0.5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.rules_count = rules
        self.in_features = in_features
        self.out_features = out_features

        self.binary = binary

        if binary:
            self.out_features = out_features = 1

        self.drop_out_p = drop_out_p

        self.device = device

        self.mean = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        self.std = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        self.literal = nn.Parameter(torch.randn(
            (in_features, rules), **factory_kwargs) * 0.1)

        self.local_slopes = nn.Parameter(torch.randn((rules, in_features, out_features), **factory_kwargs) * 0.01)
        self.local_biases = nn.Parameter(torch.zeros((rules, out_features), **factory_kwargs))

        self.decoder_linear = nn.Linear(
            in_features=rules, out_features=in_features, bias=True, **factory_kwargs)

        self.sigmoid = nn.Sigmoid()
        # Kept for constructor compatibility; not applied to phi (paper has no phi dropout).
        self.drop_out = nn.Dropout(p=drop_out_p)

        self.temp = nn.Parameter(torch.randn(
            (in_features, rules), **factory_kwargs) * 0.1)
        self.comb_weight = nn.Parameter(torch.randn((in_features, rules), **factory_kwargs) * 0.1)

    @staticmethod
    def _binary_shannon_entropy(weight: torch.Tensor) -> torch.Tensor:
        """Full Bernoulli Shannon entropy in nats: H(w) = -w log w - (1-w) log(1-w)."""
        eps = 1e-10
        safe_weight = weight.clamp(eps, 1.0 - eps)
        return -(
            safe_weight * torch.log(safe_weight)
            + (1.0 - safe_weight) * torch.log(1.0 - safe_weight)
        )

    def forward(self, X):
        phi = self.encode(X)
        reconstructed_X = self.decoder_linear(phi)
        logits = self.tsk(X, phi)
        return logits, reconstructed_X

    def encode(self, X):
        mean = self.mean.view(1, *self.mean.shape)

        sigma = F.softplus(self.std).clamp_min(1e-3).view(1, *self.std.shape)

        X = X.view(*X.shape, 1)

        def gaussmf(x, mu, sigma_):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma_ ** 2))

        # Gaussian membership with negation (equality / inequality)
        mu_pos = gaussmf(X, mean, sigma)
        literal = self.sigmoid(self.literal)
        mu_pos_neg = (mu_pos * literal) + (1 - mu_pos) * (1 - literal)

        # Sigmoidal at-least / at-most using the same positive sigma
        mu_greater = torch.sigmoid((X - mean) * sigma)
        temp = self.sigmoid(self.temp)
        mu_great_less = (mu_greater * temp) + (1 - mu_greater) * (1 - temp)

        weight = torch.sigmoid(self.comb_weight)            # (in_features, rules)
        weight = weight.unsqueeze(0)                        # (1, in_features, rules)
        mu = weight * mu_pos_neg + (1 - weight) * mu_great_less

        # Product T-norm in log-space, then normalize across rules
        eps = 1e-10
        log_firing = torch.log(mu.clamp_min(eps)).sum(dim=1)
        phi = torch.softmax(log_firing, dim=1)
        return phi

    def tsk(self, X, y):
        """
        Beta-weighted Entropy-Relaxed TSK consequent.

        Architecture
        ────────────

            α₁ = sigmoid(literal)      participation weight of the Gaussian
                                       (equality) branch and its negation
            α₂ = sigmoid(temp)         participation weight of the at-least /
                                       at-most (sigmoidal) branch

            β  = sigmoid(comb_weight)  mixing coefficient used in encode():
                                           μ = β·μ_pos_neg + (1-β)·μ_great_less
                                       β plays the same role here: it weights
                                       the two entropy contributions.

        Entropy terms  (full Bernoulli Shannon entropy, nats)
        ─────────────────────────────────────────────────────
            H₁ = -α₁ log α₁ - (1-α₁) log(1-α₁)
            H₂ = -α₂ log α₂ - (1-α₂) log(1-α₂)


        Per-feature, per-rule relaxation
        ─────────────────────────────────
            ρ_{i,j} = β · H₁_{i,j} + (1-β) · H₂_{i,j}


        TSK output
        ──────────
            y_i = Σ_j [ (1 - ρ_{i,j}) · a_{i,j} · (x_j - m_{i,j}) / σ_{i,j} ] + b_i

        where σ_{i,j} = softplus(std) > 0 is the shared width parameter.

        Shapes
        ──────
            X      : [B, F]
            y      : [B, R]   (normalised firing strengths from encode)
            output : [B, O]
        """
        eps = 1e-10

        X64      = X.double()
        means64  = self.mean.double()

        sigma64  = F.softplus(self.std).clamp_min(1e-3).double()
        slopes64 = self.local_slopes.double()
        biases64 = self.local_biases.double()
        y64      = y.double()

        alpha1 = torch.sigmoid(self.literal).double()       # Gaussian branch weight
        alpha2 = torch.sigmoid(self.temp).double()          # Sigmoidal branch weight
        beta   = torch.sigmoid(self.comb_weight).double()   # mixing coefficient

        H1 = self._binary_shannon_entropy(alpha1)
        H2 = self._binary_shannon_entropy(alpha2)

        # relaxation term  ρ_{i,j} = β·H₁ + (1-β)·H₂
        rho = beta * H1 + (1.0 - beta) * H2

        # gate: (1 - ρ), reshaped for broadcasting
        gate = (1.0 - rho)
        gate = gate.T.unsqueeze(-1)

        sigma_rs = sigma64.T.unsqueeze(-1)

        # relaxed & scaled slopes: a_{i,j} · (1 - ρ_{i,j}) / σ_{i,j}
        slopes_relaxed = gate * slopes64 / (sigma_rs + eps)

        # shifted inputs: (x_j - m_{i,j})
        X_exp     = X64.unsqueeze(1).unsqueeze(3)
        means_exp = means64.T.unsqueeze(0).unsqueeze(3)
        shifted   = X_exp - means_exp

        # linear combination over features
        slopes_exp   = slopes_relaxed.unsqueeze(0)
        linear_terms = torch.matmul(
            shifted.transpose(-2, -1), slopes_exp
        ).squeeze(-2)

        # add bias
        rule_outputs = linear_terms + biases64.unsqueeze(0)

        y64_r  = y64.reshape(-1, self.rules_count, 1)
        result = (rule_outputs * y64_r).sum(dim=1)

        return result.to(X.dtype)

    def get_interpretable_params(self):
        with torch.no_grad():
            literal = torch.sigmoid(self.literal)   # α₁
            temp    = torch.sigmoid(self.temp)       # α₂
            beta    = torch.sigmoid(self.comb_weight)  # β

            H1 = self._binary_shannon_entropy(literal)
            H2 = self._binary_shannon_entropy(temp)
            rho = beta * H1 + (1.0 - beta) * H2

            stats = {
                # α₁ - Gaussian branch participation
                "alpha1_mean": literal.mean().item(),
                "alpha1_std":  literal.std().item(),
                "alpha1_saturation": ((literal < 0.1) | (literal > 0.9)).float().mean().item(),
                # α₂ - Sigmoidal branch participation
                "alpha2_mean": temp.mean().item(),
                "alpha2_std":  temp.std().item(),
                "alpha2_saturation": ((temp < 0.1) | (temp > 0.9)).float().mean().item(),
                # β - mixing / weighting coefficient
                "beta_mean": beta.mean().item(),
                "beta_std":  beta.std().item(),
                "beta_saturation": ((beta < 0.1) | (beta > 0.9)).float().mean().item(),
                # Relaxation ρ_{i,j} diagnostics
                "relaxation_mean": rho.mean().item(),
                "relaxation_std":  rho.std().item(),
                "relaxation_high": (rho > 0.8).float().mean().item(),  # heavily relaxed features
                "relaxation_low":  (rho < 0.2).float().mean().item(),  # fully active features
                # Consequent parameter diagnostics
                "slope_mean":  self.local_slopes.mean().item(),
                "slope_std":   self.local_slopes.std().item(),
                "bias_mean":   self.local_biases.mean().item(),
                "center_mean": self.mean.mean().item(),
                "phi_mean":    F.softplus(self.std).clamp_min(1e-3).mean().item(),
                "phi_std":     F.softplus(self.std).clamp_min(1e-3).std().item(),
            }
        return stats


class MamdaniReLaFIS(ReLaFIS):
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, 
                 drop_out_p=0.5, device=None, dtype=None):
        super().__init__(in_features, rules, out_features, binary, drop_out_p, device, dtype)
    
        factory_kwargs = {'device': device, 'dtype': dtype}
        if binary:
            self.out_features = out_features = 1
        self.mamdani_linear = nn.Linear(rules, out_features, bias=True, **factory_kwargs)

    def mamdani(self, y):
        return self.mamdani_linear(y)
    
    def forward(self, X):
        phi = self.encode(X)
        reconstructed_X = self.decoder_linear(phi)
        logits = self.mamdani(phi)
        return logits, reconstructed_X


class SklearnReLaFISWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn wrapper for ReLaFIS model
    """
    def __init__(self, model, device=None, dtype=torch.float32):
        self.device = device if device else 'cpu'
        self.dtype = dtype
        self.model = model.to(self.device)

    def fit(self, X, y):
        return self

    def predict(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        with torch.no_grad():
            y_pred = self.model(X)[0]

        if self.model.binary:
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy() > 0.5
        else:
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.argmax(dim=1).cpu().numpy()
        return y_pred

    def predict_proba(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        with torch.no_grad():
            y_pred = self.model(X)[0]
        
        if self.model.binary:
            y_pred = torch.sigmoid(y_pred)
            # Convert to (n_samples, 2) format
            neg_proba = 1 - y_pred
            y_pred = torch.cat([neg_proba, y_pred], dim=1)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
    
        return y_pred.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _convert_to_tensor(self, data):
        """ Helper function to convert numpy arrays to torch tensors and move to the correct device. """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)

        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)

        elif isinstance(data, pd.DataFrame):
            data = torch.tensor(
                data.values, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(
                "Input data must be a NumPy array or a PyTorch tensor.")
        return data

    def _check_is_filiteraled(self):
        pass

    def get_params(self, deep=True):
        return {
            'model': self.model
        }

    def set_params(self, **parameters):
        return self