import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd


class ReLaFIS(nn.Module):

    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, drop_out_p=0.5, device=None, dtype=None,
                 zeta1=1.0, zeta2=1.0, zeta3=1.0, aggregation="product"):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.rules_count = rules
        self.in_features = in_features
        self.out_features = out_features

        self.binary = binary

        self.drop_out_p = drop_out_p
        self.zeta1 = float(zeta1)
        self.zeta2 = float(zeta2)
        self.zeta3 = float(zeta3)
        if aggregation == "min":
            aggregation = "godel_min"
        if aggregation not in {"product", "godel_min"}:
            raise ValueError("aggregation must be 'product' or 'godel_min'")
        self.aggregation = aggregation

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
        phi_used = self.drop_out(phi)
        reconstructed_X = self.decoder_linear(phi_used)
        logits = self.tsk(X, phi_used)
        return logits, reconstructed_X

    def _relation_weights(self):
        return (
            torch.sigmoid(self.zeta1 * self.literal),
            torch.sigmoid(self.zeta2 * self.temp),
            torch.sigmoid(self.zeta3 * self.comb_weight),
        )

    def encode(self, X):
        mean = self.mean.view(1, *self.mean.shape)

        sigma = F.softplus(self.std).clamp_min(1e-3).view(1, *self.std.shape)

        X = X.view(*X.shape, 1)

        def gaussmf(x, mu, sigma_):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma_ ** 2))

        # Gaussian membership with negation (equality / inequality)
        mu_pos = gaussmf(X, mean, sigma)
        literal, temp, weight = self._relation_weights()
        mu_pos_neg = (mu_pos * literal) + (1 - mu_pos) * (1 - literal)

        # Sigmoidal at-least / at-most using the same positive sigma
        mu_greater = torch.sigmoid((X - mean) * sigma)
        mu_great_less = (mu_greater * temp) + (1 - mu_greater) * (1 - temp)

        weight = weight.unsqueeze(0)                        # (1, in_features, rules)
        mu = weight * mu_pos_neg + (1 - weight) * mu_great_less

        if self.aggregation == "product":
            # Product T-norm in log-space, then normalize across rules.
            log_firing = torch.log(mu.clamp_min(1e-10)).sum(dim=1)
            phi = torch.softmax(log_firing, dim=1)
        else:
            firing = mu.min(dim=1).values
            total = firing.sum(dim=1, keepdim=True)
            normalized = firing / total.clamp_min(1e-10)
            uniform = torch.full_like(firing, 1.0 / self.rules_count)
            phi = torch.where(total > 1e-10, normalized, uniform)
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

        alpha1, alpha2, beta = (weight.double() for weight in self._relation_weights())

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

    def antecedent_relation_entropy(self, per_rule=False):
        """Entropy (nats) over the four dominant antecedent relation categories."""
        from utils.linguistic_richness import categories_from_two_branch, richness_from_categories

        with torch.no_grad():
            w1, w2, w3 = self._relation_weights()
            category = categories_from_two_branch(w1, w2, w3)
            return richness_from_categories(
                category, self.rules_count, self.in_features, per_rule=per_rule
            )

    def linguistic_richness(self, per_rule=False):
        """Backward-compatible alias for antecedent relation entropy."""
        return self.antecedent_relation_entropy(per_rule=per_rule)

    def inference_parameter_count(self):
        return sum(p.numel() for name, p in self.named_parameters()
                   if p.requires_grad and not name.startswith("decoder_linear."))

    def training_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_interpretable_params(self):
        with torch.no_grad():
            literal, temp, beta = self._relation_weights()

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
                "relaxation_high": (rho > 0.8 * np.log(2.0)).float().mean().item(),
                "relaxation_low":  (rho < 0.2 * np.log(2.0)).float().mean().item(),
                # Consequent parameter diagnostics
                "slope_mean":  self.local_slopes.mean().item(),
                "slope_std":   self.local_slopes.std().item(),
                "bias_mean":   self.local_biases.mean().item(),
                "center_mean": self.mean.mean().item(),
                "sigma_mean": F.softplus(self.std).clamp_min(1e-3).mean().item(),
                "sigma_std": F.softplus(self.std).clamp_min(1e-3).std().item(),
                # Deprecated aliases retained for existing diagnostic scripts.
                "phi_mean": F.softplus(self.std).clamp_min(1e-3).mean().item(),
                "phi_std": F.softplus(self.std).clamp_min(1e-3).std().item(),
            }
        return stats


class MamdaniReLaFIS(ReLaFIS):
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, 
                 drop_out_p=0.5, device=None, dtype=None):
        super().__init__(in_features, rules, out_features, binary, drop_out_p, device, dtype)
    
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mamdani_linear = nn.Linear(rules, out_features, bias=True, **factory_kwargs)

    def mamdani(self, y):
        return self.mamdani_linear(y)
    
    def forward(self, X):
        phi = self.encode(X)
        phi_used = self.drop_out(phi)
        reconstructed_X = self.decoder_linear(phi_used)
        logits = self.mamdani(phi_used)
        return logits, reconstructed_X


class SklearnReLaFISWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn wrapper for ReLaFIS model
    """
    def __init__(self, model, device=None, dtype=torch.float32):
        self.device = device if device else 'cpu'
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=self.dtype)

    def fit(self, X, y):
        return self

    def predict(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)
        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(X)[0]

        return torch.softmax(y_pred, dim=1).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)
        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(X)[0]
        
        y_pred = torch.softmax(y_pred, dim=1)
    
        return y_pred.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _convert_to_tensor(self, data):
        """ Helper function to convert numpy arrays to torch tensors and move to the correct device. """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=self.dtype, device=self.device)

        elif isinstance(data, torch.Tensor):
            data = data.to(device=self.device, dtype=self.dtype)

        elif isinstance(data, pd.DataFrame):
            data = torch.tensor(
                data.values, dtype=self.dtype, device=self.device)
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
