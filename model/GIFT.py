import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd


class GIFT(nn.Module):
    """
    GIFT: Gaussian with Integrated Fuzzy Transformation
    Uses sigmoid for "greater than mu" and its negation for "less than mu"
    """
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, zeta: float, drop_out_p=0.5, device=None, dtype=None):
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
        self.zeta = zeta  

        self.mean = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        self.std = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        self.literal = nn.Parameter(torch.randn(
            (in_features, rules), **factory_kwargs) * 0.1)

        self.tsk_linear = nn.Linear(
            in_features=in_features, out_features=rules * out_features, bias=True, **factory_kwargs)
        self.decoder_linear = nn.Linear(
            in_features=rules, out_features=in_features, bias=True, **factory_kwargs)

        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(p=drop_out_p)

        self.sigmoid_slope = nn.Parameter(torch.ones(
            (in_features, rules), **factory_kwargs))
        self.temp = nn.Parameter(torch.randn(
            (in_features, rules), **factory_kwargs) * 0.1)
        self.combination_weights = nn.Parameter(torch.randn(
            (in_features, rules, 2), **factory_kwargs))  # 2 branches: pos_neg and great_less
        
        # Relaxation parameters
        self.relax = nn.Parameter(torch.zeros((in_features, rules), **factory_kwargs))
        self.tsk_gate = nn.Parameter(torch.zeros((rules, out_features), **factory_kwargs))
        self.tsk_zeta = zeta   # re‑use the same scaling factor for both relaxations

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
        std = self.std.view(1, *self.std.shape)
        
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

        # Learnable weighted combination using softmax
        # combination_weights shape: (batch, in_features, rules, 2)
        weights = self.combination_weights  # (in_features, rules, 2)
        weights = weights.view(1, *weights.shape)  # (1, in_features, rules, 2)
        
        weights = F.softmax(weights, dim=-1)  # (1, in_features, rules, 2)
        weight_pos_neg = weights[..., 0]  # (1, in_features, rules)
        weight_great_less = weights[..., 1]  # (1, in_features, rules)
        
        mu = (weight_pos_neg * mu_pos_neg) + (weight_great_less * mu_great_less)


        # Relaxation
        relaxer = self.sigmoid(self.relax * self.zeta)     # (in_features, rules)
        relaxer = relaxer.unsqueeze(0)                     # (1, in_features, rules)
        mu = relaxer + (1 - relaxer) * mu                  # push mu towards 1 when relaxer is high
        
        epsilon = 1e-10
        y = torch.log(mu + epsilon)

        max_log_y = torch.max(y, dim=1, keepdim=True)[0]

        y = torch.sum(y - max_log_y, dim=1)

        y = torch.exp(y) * torch.exp(max_log_y.squeeze(dim=1))
        
        return y

    def tsk(self, X, y):
        # Linear projection of inputs
        X = self.tsk_linear(X)                         # (b, rules*out)
        X = X.reshape(-1, self.rules_count, self.out_features)   # (b, r, o)
        
        # Relaxation on the consequent
        gate = torch.sigmoid(self.tsk_gate * self.tsk_zeta)   # (r, o)
        gate = gate.unsqueeze(0)                               # (1, r, o)
        X = X * (1 - gate)                                     # gate small -> keep X, gate large -> shrink X
        
        # Multiply by rule activations and sum
        y = y.reshape(-1, self.rules_count, 1)                 # (b, r, 1)
        X = X * y
        return X.sum(dim=1)


class MamdaniGIFT(GIFT):
    """
    Mamdani version of GIFT model
    """
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, drop_out_p=0.5, device=None, dtype=None):
        super().__init__(in_features, rules, out_features, binary, drop_out_p, device, dtype)
    
        factory_kwargs = {'device': device, 'dtype': dtype}

        if binary:
            self.out_features = out_features = 1

        self.mamdani_linear = nn.Linear(
            in_features=rules, out_features=out_features, bias=True, **factory_kwargs)

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


class SklearnGIFTWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn wrapper for GIFT model
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