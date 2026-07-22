import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd


class LitAnfis(nn.Module):
    # Number of distinct linguistic-relation categories on the SHARED,
    # absolute 4-category scale used across every model in this codebase
    # (equal / not-equal / greater-than / less-than — see
    # linguistic_richness_utils.py). LitAnfis has no relational
    # greater/less branch, so it structurally only ever populates 2 of
    # these 4 categories — its richness is therefore capped at log(2),
    # not because of a different scale, but because of a real
    # architectural limitation, which is exactly what makes the number
    # directly comparable against models that do reach log(4).
    N_LINGUISTIC_CATEGORIES = 4

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
        # into NaN (same failure mode fixed in GIFT).
        std = self.std.clamp(min=1e-3).view(1, *self.std.shape)

        X = X.view(*X.shape, 1)

        def gaussmf(x, mu, sigma):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        y = gaussmf(X, mean, std)

        literal = self.sigmoid(self.literal)
        y = (y * literal) + (1 - y) * (1 - literal)

        epsilon = 1e-10
        y = torch.log(y + epsilon)

        # FIX (same issue as GIFT): firing strength was the PRODUCT
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

    def linguistic_richness(self, per_rule: bool = False):
        """
        Linguistic-richness metric (LitAnfis variant) — ABSOLUTE scale.

        Unlike GIFT, LitAnfis's antecedent has only the Gaussian
        branch (`mean`, `std`, `literal`) — there is no relational
        (greater-than / less-than) branch and no β mixing coefficient.

        IMPORTANT: this is computed on the SAME shared 4-category support
        used by every model in this codebase (see
        `linguistic_richness_utils.py`), NOT on a LitAnfis-only 2-category
        support. The 4 categories are:

            0: equal
            1: not-equal
            2: greater-than
            3: less-than

        LitAnfis structurally never populates categories 2/3 (it has no
        relational branch to express them), so its per-rule distribution
        always has zero mass there. That is not a normalization choice —
        it is a real, absolute consequence of the architecture, and it is
        exactly what makes the resulting number comparable across models:
        a LitAnfis rule can score at most log(2) ≈ 0.693 on this shared
        log(4) ≈ 1.386 scale, while a GIFT/GIFTSHIFT rule that
        actually spreads across all 4 categories can score up to log(4).
        That gap IS the richness gap — no per-model renormalization needed.

        For each rule i we hard-assign every one of the n=in_features terms
        to whichever of the 2 reachable categories it dominantly expresses,
        count how many features land in each, and form the empirical
        distribution p_r = n_r / n over all 4 categories (0 for r=2,3).
        The Shannon entropy H_i = -Σ_r p_r·log(p_r) is then averaged over
        all rules.

        Returns
        -------
        float
            Mean entropy across rules (natural log / nats), in [0, log(4)],
            but structurally bounded above by log(2) ≈ 0.693 for this model
            class because categories 2/3 are unreachable.
        per_rule : bool
            If True, also return the per-rule entropy tensor of shape
            (rules_count,) alongside the scalar mean, as (mean, per_rule_H).
        """
        with torch.no_grad():
            from utils.linguistic_richness import categories_from_one_branch, richness_from_categories

            alpha = torch.sigmoid(self.literal)  # (in_features, rules) — equal vs not-equal
            category = categories_from_one_branch(alpha)  # values in {0, 1} only

            return richness_from_categories(
                category, self.rules_count, self.in_features, per_rule=per_rule
            )

    def relaxation_rate(self, per_rule: bool = False):
        """
        Relaxation rate — ABSOLUTE scale, structural zero for LitAnfis.

        LitAnfis's only antecedent gate is `literal` (alpha = sigmoid
        (literal)), which is a RELATIONAL equal/not-equal gate — it
        picks which of two relations a term expresses, exactly like
        the relational branch of GIFT. It has no separate
        "don't care" / relaxation gate (no ζ-style term that blends
        membership toward a uniform value the way UNFIS's `s` or
        GRIFFIN's `s` do). So every rule's relaxation rate is
        structurally exactly 0.0 for this architecture, reported here
        (not NaN) so it's directly comparable on the same absolute
        scale against UNFIS / GRIFFIN / GIFT.

        Returns
        -------
        float
            Always 0.0.
        per_rule : bool
            If True, also return a per-rule zero tensor of shape
            (rules_count,), as (0.0, per_rule_rate).
        """
        with torch.no_grad():
            per_rule_rate = torch.zeros(self.rules_count, dtype=torch.float64, device=self.mean.device)
        if per_rule:
            return 0.0, per_rule_rate
        return 0.0

    def get_interpretable_params(self):
        with torch.no_grad():
            literal = torch.sigmoid(self.literal)

            linguistic_richness_mean, linguistic_richness_per_rule = self.linguistic_richness(per_rule=True)
            relaxation_rate_mean, relaxation_rate_per_rule = self.relaxation_rate(per_rule=True)

            stats = {
                # Linguistic richness - entropy (nats) computed on the
                # SHARED 4-category support (equal / not-equal / greater /
                # less) used by every model class, so this number is
                # directly comparable across models. LitAnfis has no
                # relational branch, so it structurally never reaches
                # categories 2/3 and is capped at log(2) ≈ 0.693 out of
                # the shared log(4) ≈ 1.386 ceiling — that gap IS the
                # richness deficit, not a scale artifact.
                "linguistic_richness": linguistic_richness_mean,
                "linguistic_richness_per_rule_std": linguistic_richness_per_rule.std().item(),
                # Always 0.0 for LitAnfis — see relaxation_rate() docstring.
                # Reported here for direct comparison against
                # UNFIS/GRIFFIN/GIFT on the same absolute scale.
                "relaxation_rate": relaxation_rate_mean,
                "relaxation_rate_per_rule": relaxation_rate_per_rule.cpu().numpy(),
                "relaxation_rate_per_rule_std": relaxation_rate_per_rule.std().item(),
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