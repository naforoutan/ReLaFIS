import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd


class GIFTSHIFTER(nn.Module):
    """
    GIFT: Gaussian with Integrated Fuzzy Transformation
    Uses sigmoid for "greater than mu" and its negation for "less than mu"
    """
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
        self.drop_out = nn.Dropout(p=drop_out_p)

        self.sigmoid_slope = nn.Parameter(torch.ones(
            (in_features, rules), **factory_kwargs))
        self.temp = nn.Parameter(torch.randn(
            (in_features, rules), **factory_kwargs) * 0.1)
        self.comb_weight = nn.Parameter(torch.randn((in_features, rules), **factory_kwargs) * 0.1) 
        

    def forward(self, X):
        y = self.encode(X)
        entropy = - y * torch.log(y + 1e-10)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)

        y = self.drop_out(y)

        reconstructed_X = self.decoder_linear(y)

        y = self.tsk(X, y)

        return y, reconstructed_X, entropy

    def encode(self, X):
        mean = self.mean.view(1, *self.mean.shape)
        std = F.softplus(self.std).view(1, *self.std.shape)
        
        X = X.view(*X.shape, 1)

        def gaussmf(x, mu, sigma):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        # Gaussian membership with negation
        mu_pos = gaussmf(X, mean, std)
        literal = self.sigmoid(self.literal)
        mu_pos_neg = (mu_pos * literal) + (1 - mu_pos) * (1 - literal)

        # Sigmoidal "greater than mu" with negation
        def sigmoidmf(x, mu, slope):
            return self.sigmoid((x - mu) * slope)
        
        mu_greater = sigmoidmf(X, mean, self.sigmoid_slope)
        temp = self.sigmoid(self.temp)
        mu_great_less = (mu_greater * temp) + (1 - mu_greater) * (1 - temp)


        weight = torch.sigmoid(self.comb_weight)            # (in_features, rules)
        weight = weight.unsqueeze(0)                        # (1, in_features, rules)
        mu = weight * mu_pos_neg + (1 - weight) * mu_great_less

        
        epsilon = 1e-10
        y = torch.log(mu + epsilon)

        max_log_y = torch.max(y, dim=1, keepdim=True)[0]

        y = torch.sum(y - max_log_y, dim=1)

        y = torch.exp(y) * torch.exp(max_log_y.squeeze(dim=1))
        
        return y

    def tsk(self, X, y):
        """
        Entropy-Relaxed TSK consequent:

            y_i = Σ_j [ (1 - r_ij) * a_ij * ((x_j - m_ij) / φ_ij) ] + b_i

        where:
            r_ij = -⅓ Σ_{k=1..3} [ α_k·log(α_k) + (1-α_k)·log(1-α_k) ]

            α₁ = sigmoid(literal)     — literal vs negation balance
            α₂ = sigmoid(temp)        — greater-than vs less-than balance
            α₃ = sigmoid(comb_weight) — parallel node selection balance

            φ_ij = softplus(std)      — fuzziness / scale parameter
        """
        X64       = X.double()                                   # [B, F]
        means64   = self.mean.double()                           # [F, R]
        std64     = F.softplus(self.std).double()                # [F, R]  → φ_ij
        slopes64  = self.local_slopes.double()                   # [R, F, O]
        biases64  = self.local_biases.double()                   # [R, O]
        y64       = y.double()                                   # [B, R]

        eps = 1e-10

        def binary_entropy(raw_param):
            """Sigmoid-activate then compute binary entropy. Returns [F, R] float64."""
            alpha = torch.sigmoid(raw_param).double()            # [F, R]
            h = -(alpha * torch.log(alpha + eps)
                  + (1 - alpha) * torch.log(1 - alpha + eps))   # [F, R]
            return h

        h1 = binary_entropy(self.literal)      # α₁: literal/negation
        h2 = binary_entropy(self.temp)         # α₂: greater/less-than
        h3 = binary_entropy(self.comb_weight)  # α₃: node selection

        log2 = torch.tensor(2.0, dtype=torch.float64, device=X.device).log()
        r = (h1 + h2 + h3) / 3.0   # [F, R], values in [0, 1]

        relaxation = (1.0 - r)                  # [F, R]
        relaxation = relaxation.T               # [R, F]  (transpose for alignment)
        relaxation = relaxation.unsqueeze(-1)   # [R, F, 1]

        phi = std64.T.unsqueeze(-1)             # [R, F, 1]

        slopes_relaxed = relaxation * slopes64 / (phi + eps)  # [R, F, O]

        X_exp     = X64.unsqueeze(1).unsqueeze(3)              # [B, 1, F, 1]
        means_exp = means64.T.unsqueeze(0).unsqueeze(3)        # [1, R, F, 1]
        shifted   = X_exp - means_exp                          # [B, R, F, 1]

        slopes_exp   = slopes_relaxed.unsqueeze(0)             # [1, R, F, O]
        linear_terms = torch.matmul(
            shifted.transpose(-2, -1), slopes_exp              # [B, R, 1, F] × [1, R, F, O]
        ).squeeze(-2)                                          # [B, R, O]

        rule_outputs = linear_terms + biases64.unsqueeze(0)    # [B, R, O]

        y64_r  = y64.reshape(-1, self.rules_count, 1)          # [B, R, 1]
        result = (rule_outputs * y64_r).sum(dim=1)             # [B, O]

        # Cast back to the model's native dtype (usually float32)
        return result.to(X.dtype)
    

    def get_interpretable_params(self):
        with torch.no_grad():
            literal = torch.sigmoid(self.literal)
            temp = torch.sigmoid(self.temp)
            weight = torch.sigmoid(self.comb_weight)

            stats = {
                "literal_mean": literal.mean().item(),
                "literal_std": literal.std().item(),
                "temp_mean": temp.mean().item(),
                "temp_std": temp.std().item(),
                "weight_mean": weight.mean().item(),
                "weight_std": weight.std().item(),
                "literal_saturation": ((literal < 0.1) | (literal > 0.9)).float().mean().item(),
                "temp_saturation": ((temp < 0.1) | (temp > 0.9)).float().mean().item(),
                "weight_saturation": ((weight < 0.1) | (weight > 0.9)).float().mean().item(),
                # ADD THESE LINES:
                "slope_mean": self.local_slopes.mean().item(),
                "slope_std": self.local_slopes.std().item(),
                "bias_mean": self.local_biases.mean().item(),
                "center_mean": self.mean.mean().item(),
            }
        return stats


class MamdaniGIFTSHIFTER(GIFTSHIFTER):
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
        y = self.encode(X)
        entropy = - y * torch.log(y + 1e-10)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)

        reconstructed_X = self.decoder_linear(y)

        y = self.mamdani(y)

        return y, reconstructed_X, entropy


class SklearnGIFTSHIFTERWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn wrapper for GIFTSHIFTER model
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