import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd


class LitAnfis(nn.Module):
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

        self.tsk_linear = nn.Linear(
            in_features=in_features, out_features=rules * out_features, bias=True, **factory_kwargs)
        self.decoder_linear = nn.Linear(
            in_features=rules, out_features=in_features, bias=True, **factory_kwargs)

        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(p=drop_out_p)


    def forward(self, X):
        y = self.encode(X)

        # FIX Bug 2: original had `- y * torch.log(y)` with no epsilon guard,
        # causing log(0) = -inf / NaN gradients when any rule activation hits zero.
        entropy = - y * torch.log(y + 1e-10)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)

        y = self.drop_out(y)

        reconstructed_X = self.decoder_linear(y)

        y = self.tsk(X, y)

        return y, reconstructed_X, entropy

    def encode(self, X):
        mean = self.mean.view(1, *self.mean.shape)
        # Clamp away from 0 (and negative): std is unconstrained and used as
        # sigma**2 in the denominator below, so it can blow up the gradient
        # into NaN (same failure mode fixed in GIFTSHIFTER).
        std = self.std.clamp(min=1e-3).view(1, *self.std.shape)

        X = X.view(*X.shape, 1)

        def gaussmf(x, mu, sigma):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        y = gaussmf(X, mean, std)

        literal = self.sigmoid(self.literal)
        y = (y * literal) + (1 - y) * (1 - literal)

        epsilon = 1e-10
        y = torch.log(y + epsilon)

        # FIX (same issue as GIFTSHIFTER): firing strength was the PRODUCT
        # of per-feature memberships (sum in log-space) over dim=1 =
        # in_features. For high-dimensional data (e.g. 617 features on
        # Isolet), multiplying hundreds of numbers in (0, 1] makes every
        # rule's firing strength numerically degenerate (vanishingly small
        # and nearly identical across rules/inputs, even though the
        # log-sum-exp trick prevents literal underflow to 0.0). This makes
        # the firing strengths uninformative regardless of X, so the model
        # collapses to predicting a near-constant output.
        #
        # Using the geometric MEAN instead of the product (mean instead of
        # sum in log-space) keeps the same fuzzy-AND semantics but no
        # longer shrinks with in_features, so it stays discriminative
        # regardless of input dimensionality.
        max_log_y = torch.max(y, dim=1, keepdim=True)[0]

        y = torch.mean(y - max_log_y, dim=1)

        y = torch.exp(y) * torch.exp(max_log_y.squeeze(dim=1))
        
        return y

    def tsk(self, X, y):
        X = self.tsk_linear(X)

        X = X.reshape(-1, self.rules_count, self.out_features)
        y = y.reshape(-1, self.rules_count, 1)
        X = X * y

        return X.sum(dim=1)
    
    def get_interpretable_params(self):
        with torch.no_grad():
            literal = torch.sigmoid(self.literal)

            stats = {
                "literal_mean": literal.mean().item(),
                "literal_std": literal.std().item(),
                "literal_saturation": ((literal < 0.1) | (literal > 0.9)).float().mean().item(),
                "literal_matrix": literal.cpu().numpy(),
            }
        return stats
    

class MamdaniLitAnfis(LitAnfis):
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

        # FIX Bug 2: same epsilon guard as LitAnfis.forward()
        entropy = - y * torch.log(y + 1e-10)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)

        reconstructed_X = self.decoder_linear(y)

        y = self.mamdani(y)

        return y, reconstructed_X, entropy


class SklearnLitAnfisWrapper(BaseEstimator, ClassifierMixin):
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
        # FIX Bug 1: original was `self.predict(X)[0]` which indexes the first
        # element of the numpy array instead of using the full prediction array.
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _convert_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)
        elif isinstance(data, pd.DataFrame):
            data = torch.tensor(data.values, dtype=torch.float32, device=self.device)
        else:
            raise ValueError("Input data must be a NumPy array or a PyTorch tensor.")
        return data

    def _check_is_filiteraled(self):
        pass

    def get_params(self, deep=True):
        return {'model': self.model}

    def set_params(self, **parameters):
        return self