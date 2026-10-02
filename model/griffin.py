import torch
from torch import nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score 
import pandas as pd
 
class GRIFFIN(nn.Module):
    # Number of distinct linguistic-relation categories on the SHARED,
    # absolute scale used across every model in this codebase (equal /
    # not-equal / greater-than / less-than — see
    # linguistic_richness_utils.py). GRIFFIN's `s` parameter plays the
    # same role as LitAnfis's `literal` (an equal/not-equal gate via
    # sigmoid(eta*s)) — there is no greater/less relational branch, so
    # GRIFFIN structurally only ever populates 2 of these 4 categories,
    # same as LitAnfis. Its `relaxer` term (sigmoid(s*zeta)) blends
    # toward a uniform "don't care" value but does not introduce a third
    # category — it softens membership, it doesn't change which relation
    # is being expressed.
    N_LINGUISTIC_CATEGORIES = 4

    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool = False, rank: int = 2,
                 regression: bool = False, zeta: float = 1.0, Xi: float = 1.0, eta: float = 1.0,
                 drop_out_p: float = 0.0, device=None, dtype=None):
        super().__init__()
        factory_kwargs = self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.rules_count = rules
        self.in_features = in_features
        self.out_features = out_features
        self.regression = regression
        self.binary = binary
        if binary:
            self.out_features = out_features = 1
        self.rank = rank
        self.device = device
        self.zeta = zeta
        self.eta = eta
        self.Xi = Xi
        self.drop_out_p = drop_out_p
        self.parameter_selector = nn.Parameter(torch.zeros((1, in_features), **factory_kwargs))
        self.s = nn.Parameter(torch.zeros((1, rules, rank), **factory_kwargs))
        self.V = nn.Parameter(torch.ones((1, rules, in_features, rank), **factory_kwargs))
        
        self.sigmoid = nn.Sigmoid()
        
        self.mean = nn.Parameter(torch.rand(
            (1, rules, in_features), **factory_kwargs))
        self.std = nn.Parameter(torch.rand(
            (1, rules, rank), **factory_kwargs))
        self.tsk_parameter = nn.Parameter(torch.rand(
            (1, rules, rank, out_features), **factory_kwargs))

        self.tsk_inner_bias = nn.Parameter(torch.rand((1, rules, out_features), **factory_kwargs))
        self.tsk_final_bias = nn.Parameter(torch.rand((1, out_features), **factory_kwargs))

        # Decoder for reconstruction (maps rule activations back to input space)
        self.decoder_linear = nn.Linear(rules, in_features, bias=True, **factory_kwargs)
        self.dropout = nn.Dropout(p=drop_out_p)
        self._inference_cache = None

    def invalidate_inference_cache(self):
        self._inference_cache = None

    def train(self, mode: bool = True):
        if mode:
            self.invalidate_inference_cache()
        return super().train(mode)

    def _apply(self, fn):
        self.invalidate_inference_cache()
        return super()._apply(fn)

    def _load_from_state_dict(self, *args, **kwargs):
        self.invalidate_inference_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def prepare_inference_cache(self):
        """Cache exact parameter-only transforms used by GRIFFIN inference."""
        if self.training:
            raise RuntimeError("prepare_inference_cache() requires model.eval()")
        with torch.no_grad():
            literal = torch.sigmoid(self.eta * self.s)
            relaxer = torch.sigmoid(self.s * self.zeta)
            tsk_zeta = torch.sigmoid(self.s / 4)
            self._inference_cache = {
                "selector": torch.sigmoid(self.parameter_selector * self.Xi).detach(),
                "std": self.std.clamp(min=1e-3).detach(),
                "literal": literal.detach(),
                "one_minus_literal": (1 - literal).detach(),
                "relaxer": relaxer.detach(),
                "one_minus_relaxer": (1 - relaxer).detach(),
                "one_minus_tsk_zeta": (1 - tsk_zeta).detach(),
            }
        return self

    def encode_cached(self, X):
        """Existing encode equations with parameter-only transforms cached."""
        if self._inference_cache is None:
            raise RuntimeError("Call model.eval().prepare_inference_cache() first")
        cache = self._inference_cache
        X = X * cache["selector"]
        X = torch.unsqueeze(X, dim=1)
        Z = X - self.mean
        Z = torch.unsqueeze(Z, dim=2)
        Z = torch.matmul(Z, self.V)
        Z = torch.squeeze(Z, dim=2)
        y = torch.exp(-0.5 * torch.pow(Z / cache["std"], exponent=2))
        literal = cache["literal"]
        y = (y * literal) + (1 - y) * cache["one_minus_literal"]
        relaxer = cache["relaxer"]
        y = relaxer + cache["one_minus_relaxer"] * y
        return Z, torch.clamp(torch.prod(y, dim=2), min=1e-12)

    def tsk_cached(self, Z, y):
        """Existing TSK equations with its parameter-only zeta cached."""
        if self._inference_cache is None:
            raise RuntimeError("Call model.eval().prepare_inference_cache() first")
        Z = self._inference_cache["one_minus_tsk_zeta"] * Z
        Z = torch.unsqueeze(Z, dim=2)
        Z = torch.matmul(Z, self.tsk_parameter)
        Z = torch.squeeze(Z, dim=2)
        Z = Z + self.tsk_inner_bias
        y = torch.unsqueeze(y, dim=2)
        Z = Z * y
        return Z.sum(dim=1) + self.tsk_final_bias

    def encode(self, X):
        selector = torch.sigmoid(self.parameter_selector*self.Xi) # 1,f
        X = X * selector # b,f
        
        X = torch.unsqueeze(X, dim=1) #b, 1, f
        Z = X - self.mean #(b, 1, f) - (1, r, f) -> (b, r, f)
        Z = torch.unsqueeze(Z, dim=2) # (b, r, f) -> (b, r, 1, f) 
        Z = torch.matmul(Z, self.V) # (1, r, f, f) -> (b, r, 1, f)
        Z = torch.squeeze(Z, dim=2) # (b, r, f)

        # Clamp std away from 0 (and negative): it's used as a raw divisor
        # below and is otherwise unconstrained during training, so it can
        # drift toward 0 and blow up the gradient into NaN (same failure
        # mode fixed in ReLaFIS / LitAnfis / UNFIS / ANFIS).
        std = self.std.clamp(min=1e-3)
        y = torch.exp(-0.5 * torch.pow(Z / std, exponent=2)) # member function

        epsilon = 1e-12  
        literal = self.sigmoid(self.eta * self.s)     
        y = (y * literal) + (1 - y) * (1 - literal)

        relaxer = self.sigmoid(self.s * self.zeta)    
        y = relaxer + (1 - relaxer) * y # (b, r, rank)
        
        y = torch.clamp(torch.prod(y,dim = 2) , min = epsilon)

        return Z, y

    def forward(self, X):
        Z, y_raw = self.encode(X)  # y_raw: rule activations before normalization

        # Compute entropy on raw activations (as in GIFT)
        entropy = -y_raw * torch.log(y_raw + 1e-12)  # (b, r)

        # Normalize rule activations (original GRIFFIN logic)
        if self.rules_count > 1 and not self.regression:
            y_norm = F.normalize(y_raw, p=1, dim=1)
        else:
            y_norm = y_raw

        y_norm = self.dropout(y_norm)

        # Reconstruct input from normalized rule activations
        reconstructed = self.decoder_linear(y_norm)  # (b, in_features)

        # TSK output (uses normalized activations)
        output = self.tsk(Z, y_norm)

        return output, reconstructed, entropy

    def linguistic_richness(self, per_rule: bool = False):
        """
        Linguistic-richness metric (GRIFFIN variant) — ABSOLUTE scale.

        GRIFFIN's antecedent gate `s` (shape (1, rules, rank)) plays the
        same role as LitAnfis's `literal`: an equal/not-equal gate via

            literal = sigmoid(eta * s)
            y = (y * literal) + (1 - y) * (1 - literal)

        There is no separate greater-than/less-than relational branch
        (the `relaxer = sigmoid(s * zeta)` term that follows only blends
        membership toward a uniform "don't care" value — it doesn't
        change WHICH relation a term expresses, just how strongly). So,
        like LitAnfis, GRIFFIN structurally only ever populates 2 of the
        4 categories on the SHARED scale used across this codebase (see
        `linguistic_richness_utils.py`):

            0: equal      (literal ≥ 0.5)
            1: not-equal  (literal <  0.5)

        Unlike LitAnfis (which gates over `in_features` raw features),
        GRIFFIN's gate operates over the `rank` rotated/projected
        components per rule (after the V rotation), since that's the
        space `s` actually lives in. So here the "n linguistic terms per
        rule" is `rank`, not `in_features`.

        Returns
        -------
        float
            Mean entropy across rules (nats), in [0, log(4)], but
            structurally bounded above by log(2) ≈ 0.693 for this model
            class because categories 2/3 are unreachable.
        per_rule : bool
            If True, also return the per-rule entropy tensor of shape
            (rules_count,) alongside the scalar mean, as (mean, per_rule_H).
        """
        with torch.no_grad():
            from utils.linguistic_richness import categories_from_one_branch, richness_from_categories

            # self.s: (1, rules, rank) -> squeeze to (rules, rank), then
            # transpose to (rank, rules) to match the (n_terms, rules_count)
            # convention used by richness_from_categories / bincount-per-column.
            literal = torch.sigmoid(self.eta * self.s).squeeze(0).T  # (rank, rules)
            category = categories_from_one_branch(literal)  # values in {0, 1}

            return richness_from_categories(
                category, self.rules_count, self.rank, per_rule=per_rule
            )

    def relaxation_rate(self, per_rule: bool = False):
        """
        Relaxation rate — how much each rule leans on GRIFFIN's
        `relaxer` gate, which blends membership toward a uniform
        "don't care" value:

            relaxer = sigmoid(s * zeta)
            y = relaxer + (1 - relaxer) * y

        As relaxer -> 1 a component is fully relaxed (y -> 1
        regardless of x); as relaxer -> 0 it's fully committed to the
        raw membership (after the equal/not-equal `literal` gate).
        This is the same kind of "don't care" mechanism as UNFIS's
        zeta, distinct from the relational equal/not-equal gate
        (`literal` here, `literal` in LitAnfis) which has no notion of
        relaxation at all.

        `s` (and hence `relaxer`) lives in (1, rules, rank) — the
        rotated/projected component space, same convention used in
        linguistic_richness() — so here "n terms per rule" is `rank`,
        not `in_features`. For rule r:

            rate_r = (1/rank) * sum_k relaxer_{r,k}

        and the overall scalar is the mean of rate_r over all rules.

        Returns
        -------
        float
            Mean relaxation rate across rules, in [0, 1].
        per_rule : bool
            If True, also return the per-rule rate tensor of shape
            (rules_count,), as (mean, per_rule_rate).
        """
        with torch.no_grad():
            relaxer = torch.sigmoid(self.s * self.zeta).squeeze(0)  # (rules, rank)
            per_rule_rate = relaxer.mean(dim=1)  # sum_k relaxer / rank, shape (rules,)
            mean_rate = per_rule_rate.mean().item()

        if per_rule:
            return mean_rate, per_rule_rate
        return mean_rate

    def get_interpretable_params(self):
        with torch.no_grad():
            relaxation_rate_mean, relaxation_rate_per_rule = self.relaxation_rate(per_rule=True)

            stats = {
                "relaxation_rate": relaxation_rate_mean,
                "relaxation_rate_per_rule": relaxation_rate_per_rule.cpu().numpy(),
                "relaxation_rate_per_rule_std": relaxation_rate_per_rule.std().item(),
            }
        return stats

    def tsk(self, Z, y):
        # Z shape -> b, r, rank
        zeta = self.sigmoid(self.s / 4)  # 1, r, rank
        Z = (1 - zeta) * Z
        Z = torch.unsqueeze(Z, dim=2)  # (b, r, 1, rank)
        Z = torch.matmul(Z, self.tsk_parameter)  # (b, r, 1, o)
        Z = torch.squeeze(Z, dim=2)  # (b, r, o)
        Z = Z + self.tsk_inner_bias
        y = torch.unsqueeze(y, dim=2)  # b, r, 1
        Z = Z * y
        return Z.sum(dim=1) + self.tsk_final_bias

    def _init(self, X_train, y_train): 
        gmm = GaussianMixture(n_components=self.rules_count, covariance_type='full', random_state=24)
        X = np.concatenate([X_train, y_train[..., None]], axis=1)
        gmm.fit(X)  
        labels = gmm.predict(X)
        labels = torch.from_numpy(labels)
        X_train = torch.from_numpy(X_train)
        
        
        V = torch.ones(1, self.rules_count, self.in_features, self.rank)        
        mean = torch.rand((1, self.rules_count, self.in_features))
        std = torch.rand((1, self.rules_count, self.rank))

        for i in range(self.rules_count):
            cluster_points = X_train[labels == i]
            if cluster_points.shape[0] > 1:
                
                m = torch.mean(cluster_points, dim=0, keepdims=True)
                mean[:, i] = m
                cluster_points = cluster_points - m
                
                cov = cluster_points.T @ cluster_points
                eigL, eigV = torch.linalg.eig(cov)
                eigL = eigL.real
                eigV = eigV.real

                topk = torch.topk(eigL, k=self.rank)
                idx = topk.indices

                eigL = eigL[idx]
                eigV = eigV[:, idx]

                V[0, i, :, :self.rank] = eigV
                std[0, i, :self.rank] = eigL

                
        std[torch.abs(std) < 1e-9] = 1.0
        self.V = nn.Parameter(V)
        self.mean = nn.Parameter(mean)
        self.std = nn.Parameter(std)

        
    def _init_fexmax(self, X_train: np.ndarray, threshold: float = 1e-6, max_iter: int = 50):
        X = torch.from_numpy(X_train).float().to(self.device)
        N, d = X.shape
        R = self.rules_count
        M = torch.rand((R, d), device=self.device)
        Gamma = torch.eye(d, device=self.device).unsqueeze(0).repeat(R, 1, 1)
        Delta = torch.eye(d, device=self.device).unsqueeze(0).repeat(R, 1, 1)
        for it in range(max_iter):
            mu = []
            for i in range(R):
                diff = X - M[i]
                Q_inv = torch.inverse(Gamma[i] @ Delta[i] @ Gamma[i].T + 1e-6*torch.eye(d, device=self.device))
                exponent = -0.5 * torch.sum(diff @ Q_inv * diff, dim=1)
                mu.append(torch.exp(exponent))
            mu = torch.stack(mu, dim=1)
            mu = mu / (mu.sum(dim=1, keepdim=True) + 1e-12)
            for i in range(R):
                weights = mu[:, i].unsqueeze(1)
                M[i] = (weights * X).sum(dim=0) / (weights.sum() + 1e-12)
                diff = X - M[i]
                cov = (weights * diff).T @ diff / (weights.sum() + 1e-12)
                eigvals, eigvecs = torch.linalg.eigh(cov)
                mask = eigvals > threshold
                eigvals = eigvals[mask]
                eigvecs = eigvecs[:, mask]
                Gamma[i] = eigvecs
                Delta[i] = torch.diag(eigvals)
        self.mean = nn.Parameter(M.unsqueeze(0))
        self.V = nn.Parameter(Gamma.unsqueeze(0))
        self.std = nn.Parameter(torch.stack([torch.diag(Delta[i]) for i in range(R)]).unsqueeze(0))


class SklearnGRIFFINrapper(BaseEstimator, ClassifierMixin):
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
            y_pred, _, _ = self.model(X)
        if self.model.out_features == 1 and not self.model.regression:
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy() > 0.5
        elif self.model.regression:
            return y_pred
        else:
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.argmax(dim=1).cpu().numpy()
        return y_pred

    def predict_proba(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)
        with torch.no_grad():
            y_pred, _, _ = self.model(X)
        if self.model.out_features == 1:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
        return y_pred.cpu().numpy()

    def score(self, X, y):
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
