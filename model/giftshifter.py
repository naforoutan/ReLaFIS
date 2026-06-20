import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd


class GIFTSHIFTER(nn.Module):

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
        # Clamp softplus(std) away from 0. softplus(std) can underflow to exactly
        # 0.0 in float32 once `std` drifts to large negative values (nothing
        # constrains it during training). Since std appears as sigma**2 in the
        # denominator of the Gaussian membership function below, an
        # unclamped near-zero sigma causes the forward value and gradient to
        # blow up, which pushes std even further negative — a runaway
        # feedback loop that crashes the whole model to NaN within a few
        # steps. A real floor (not just a tiny epsilon) breaks that loop.
        std = F.softplus(self.std).clamp(min=1e-3).view(1, *self.std.shape)
        
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

        # FIX: firing strength was the PRODUCT of per-feature memberships
        # (sum in log-space) over dim=1 = in_features. For high-dimensional
        # data (e.g. 617 features on Isolet), multiplying 617 numbers in
        # (0, 1] makes every rule's firing strength numerically degenerate
        # (vanishingly small and nearly identical across rules/inputs even
        # though log-sum-exp prevents literal underflow to 0.0). This made
        # the firing strengths uninformative regardless of X, so the model
        # collapsed to predicting a near-constant output.
        #
        # Using the geometric MEAN instead of the product (mean instead of
        # sum in log-space) keeps the same fuzzy-AND semantics — still
        # driven by how well X matches every feature's membership — but the
        # result no longer shrinks with in_features, so it stays
        # discriminative regardless of input dimensionality.
        max_log_y = torch.max(y, dim=1, keepdim=True)[0]

        y = torch.mean(y - max_log_y, dim=1)

        y = torch.exp(y) * torch.exp(max_log_y.squeeze(dim=1))
        
        return y

    def tsk(self, X, y):
        """
        Beta-weighted Entropy-Relaxed TSK consequent.

        Architecture
        ────────────
        The antecedent has two parallel branches whose weights are the α's:

            α₁ = sigmoid(literal)      participation weight of the Gaussian
                                       (equality) branch and its negation
            α₂ = sigmoid(temp)         participation weight of the Less-than /
                                       Greater-than (sigmoidal) branch

            β  = sigmoid(comb_weight)  mixing coefficient used in encode():
                                           μ = β·μ_pos_neg + (1-β)·μ_great_less
                                       β plays the same role here: it weights
                                       the two entropy contributions.

        Entropy terms  (one-sided: measures how "committed" each α is)
        ──────────────────────────────────────────────────────────────
            H₁ = -α₁ · log(α₁)        ∈ [0, 1/e]  max at α=1/e ≈ 0.368
            H₂ = -α₂ · log(α₂)        ∈ [0, 1/e]

        Note: one-sided entropy (not binary entropy) is used deliberately.
        It is highest when α is small (uncertain / near 0) and collapses to 0
        when α → 1 (fully committed). This matches the semantic: a small α
        means the branch barely participates, so its contribution should be
        relaxed away.

        Normalisation: divide by max value 1/e so r_{i,j} ∈ [0, 1].

        Per-feature, per-rule relaxation
        ─────────────────────────────────
            r_{i,j} = β · H₁_{i,j} + (1-β) · H₂_{i,j}

        β mirrors its role in encode(): when β→1 the Gaussian branch dominates
        and its entropy H₁ drives relaxation; when β→0 the sigmoidal branch
        dominates and H₂ drives relaxation.

        TSK output
        ──────────
            y_i = Σ_j [ (1 - r_{i,j}) · a_{i,j} · (x_j - m_{i,j}) / φ_{i,j} ] + b_i

        where φ_{i,j} = softplus(std) > 0 is the fuzziness scale.

        All intermediates use float64 for numerical stability; output is cast
        back to the model's native dtype.

        Shapes
        ──────
            X      : [B, F]
            y      : [B, R]   (normalised firing strengths from encode)
            output : [B, O]
        """
        eps = 1e-10

        X64      = X.double()
        means64  = self.mean.double() 
        # Same floor as encode(): softplus(std) can underflow to 0 in float32,
        # and phi64 is used as a divisor below, so an unclamped near-zero
        # value causes exploding gradients that drive std further negative
        # (runaway → NaN). Match the clamp used in encode() for consistency.
        phi64    = F.softplus(self.std).clamp(min=1e-3).double()
        slopes64 = self.local_slopes.double()
        biases64 = self.local_biases.double()
        y64      = y.double()

        alpha1 = torch.sigmoid(self.literal).double()       # Gaussian branch weight
        alpha2 = torch.sigmoid(self.temp).double()          # Sigmoidal branch weight
        beta   = torch.sigmoid(self.comb_weight).double()   # mixing coefficient

        # one-sided entropy  H_k = -α_k · log(α_k)
        # Normalise by 1/e (the maximum of -α·log(α) on (0,1]) so r ∈ [0,1]
        one_over_e = torch.tensor(1.0 / torch.e, dtype=torch.float64, device=X.device)

        H1 = -(alpha1 * torch.log(alpha1 + eps)) / one_over_e
        H2 = -(alpha2 * torch.log(alpha2 + eps)) / one_over_e

        # relaxation term  r_{i,j} = β·H₁ + (1-β)·H₂  ∈ [0, 1]
        r = beta * H1 + (1.0 - beta) * H2 

        # gate: (1 - r), reshaped for broadcasting
        gate = (1.0 - r)
        gate = gate.T.unsqueeze(-1)

        phi_rs = phi64.T.unsqueeze(-1)

        # relaxed & scaled slopes: a_{i,j} · (1 - r_{i,j}) / φ_{i,j}
        slopes_relaxed = gate * slopes64 / (phi_rs + eps)

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

            eps = 1e-10
            one_over_e = 1.0 / torch.e
            H1 = -(literal * torch.log(literal + eps)) / one_over_e 
            H2 = -(temp    * torch.log(temp    + eps)) / one_over_e 
            r  = beta * H1 + (1.0 - beta) * H2 

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
                # Relaxation r_{i,j} diagnostics
                "relaxation_mean": r.mean().item(),
                "relaxation_std":  r.std().item(),
                "relaxation_high": (r > 0.8).float().mean().item(),  # heavily relaxed features
                "relaxation_low":  (r < 0.2).float().mean().item(),  # fully active features
                # Consequent parameter diagnostics
                "slope_mean":  self.local_slopes.mean().item(),
                "slope_std":   self.local_slopes.std().item(),
                "bias_mean":   self.local_biases.mean().item(),
                "center_mean": self.mean.mean().item(),
                "phi_mean":    F.softplus(self.std).clamp(min=1e-3).mean().item(),
                "phi_std":     F.softplus(self.std).clamp(min=1e-3).std().item(),
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