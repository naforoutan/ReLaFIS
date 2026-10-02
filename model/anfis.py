import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score  # or any other metric
import pandas as pd



class ANFIS(nn.Module):
    # Number of distinct linguistic-relation categories on the SHARED,
    # absolute scale used across every model in this codebase (equal /
    # not-equal / greater-than / less-than — see
    # linguistic_richness_utils.py). Kept at 4 for comparability even
    # though ANFIS structurally never leaves category 0 (see
    # linguistic_richness() below) — classical ANFIS has no relational
    # gate of any kind, just a plain Gaussian membership function.
    N_LINGUISTIC_CATEGORIES = 4

    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool = False,
                 drop_out_p: float = 0.5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = self.factory_kwargs = {'device': device, 'dtype': dtype}
        
        
        self.binary = binary

        self.rules_count = rules
        self.in_features = in_features
        self.out_features = out_features

        self.drop_out_p = drop_out_p

        self.device = device

        self.mean = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        self.std = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        
        
        if binary:
            self.out_features = out_features = 1


        self.mamdani_linear = nn.Linear(
            in_features=rules, out_features=out_features, bias=True, **factory_kwargs)
        self.decoder_linear = nn.Linear(
            in_features=rules, out_features=in_features, bias=True, **factory_kwargs)

        self.drop_out = nn.Dropout(p=drop_out_p)


    def encode(self, X):

        mean = self.mean.view(1, *self.mean.shape)
        # Clamp std away from 0: it's unconstrained and used as sigma**2 in
        # the denominator below, so it can drift and blow up the gradient
        # into NaN (same failure mode fixed in ReLaFIS / LitAnfis / UNFIS).
        std = self.std.clamp(min=1e-3).view(1, *self.std.shape)

        X = X.view(*X.shape, 1)

        def gaussmf(x, mu, sigma):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        y = gaussmf(X, mean, std)
        
        epsilon = 1e-1      
        
        y = torch.log(y + epsilon)

        max_log_y= torch.max(y, dim=1, keepdim=True)[0]

        # FIX (same issue as ReLaFIS / LitAnfis / GIFTSHIFT): firing
        # strength was the PRODUCT of per-feature memberships (sum in
        # log-space) over dim=1 = in_features. For high-dimensional data
        # (e.g. 617 features on Isolet), multiplying hundreds of numbers in
        # (0, 1] makes every rule's firing strength numerically degenerate
        # (vanishingly small and nearly identical across rules/inputs, even
        # though the log-sum-exp trick prevents literal underflow to 0.0).
        # This makes the firing strengths uninformative regardless of X, so
        # the model collapses to predicting a near-constant output.
        #
        # Using the geometric MEAN instead of the product (mean instead of
        # sum in log-space) keeps the same fuzzy-AND semantics but no
        # longer shrinks with in_features, so it stays discriminative
        # regardless of input dimensionality.
        y = torch.mean(y - max_log_y, dim=1)

        y = torch.exp(y) * torch.exp(max_log_y.squeeze(dim=1))
        
        return y
    
    def mamdani(self, y):
        return self.mamdani_linear(y)

    def linguistic_richness(self, per_rule: bool = False):
        """
        Linguistic-richness metric — ABSOLUTE scale, structural zero.

        Classical ANFIS has only a plain Gaussian membership function
        (`mean`, `std`) — no equal/not-equal gate (`literal`), no
        greater/less gate (`temp`), no mixing coefficient (`comb_weight`).
        Every linguistic term therefore always expresses the SAME single
        category ("equal to mu") on the shared 4-category support used
        across this codebase (see `linguistic_richness_utils.py`).

        Since every (feature, rule) entry falls into exactly one category
        with probability 1, the empirical distribution per rule is a
        one-hot vector and its Shannon entropy is exactly 0 — not because
        of a degenerate or undertrained model, but because the
        architecture has no relational vocabulary to vary across. This is
        reported as a real 0.0 (not NaN) so it can be plotted directly
        alongside richer models on the same absolute scale.

        Returns
        -------
        float
            Always 0.0.
        per_rule : bool
            If True, also return a per-rule zero tensor of shape
            (rules_count,), as (0.0, per_rule_H).
        """
        with torch.no_grad():
            entropies = torch.zeros(self.rules_count, dtype=torch.float64, device=self.mean.device)
        if per_rule:
            return 0.0, entropies
        return 0.0

    def forward(self, X):
        y = self.encode(X)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)  

        y = self.drop_out(y)

        reconstructed_X = self.decoder_linear(y)

        y = self.mamdani(y)

        return y, reconstructed_X


class SklearnAnfisWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device=None, dtype=torch.float32):
        self.device = device if device else 'cpu'
        self.dtype = dtype
        # Initialize the model
        self.model = model.to(self.device)

    def fit(self, X, y):
        return self

    def predict(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        # Use the model to get predictions
        with torch.no_grad():
            y_pred= self.model(X)[0]

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

    
if __name__ == "__main__":
    anfis = ANFIS(3, 5, 2)
    X = torch.randn(100, 3)
    y, recon = anfis(X)
    print(y.shape, recon.shape)