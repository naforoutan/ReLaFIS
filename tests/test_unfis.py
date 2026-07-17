"""Focused unit tests for the UNFIS-c PyTorch reproduction."""

from __future__ import annotations

from copy import deepcopy
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.admtsk import ADMTSK, SklearnADMTSKWrapper
from model.anfis import ANFIS, SklearnAnfisWrapper
from model.unfis import (
    UNFIS,
    SklearnUNFISWrapper,
    _build_probability_jacobian,
    _solve_damped_normal,
    gqlm_train_unfis,
    knn_initialize_unfis,
    train_unfis_model,
)
from tests.evaluate import Evaluator


class _TinyExperiment:
    """Minimal experiment stub for Evaluator smoke tests."""

    def __init__(self, n_samples=40, n_features=4, seed=0, n_folds=2, kfold=False):
        rng = np.random.RandomState(seed)
        self._X = rng.randn(n_samples, n_features).astype(np.float32)
        self._y = (self._X[:, 0] > 0).astype(np.int64)
        self.n_folds = int(n_folds)
        self._fold = 0
        self._kfold = bool(kfold)

    def is_kfold(self):
        return self._kfold

    def has_validation_split(self):
        return False

    def resplit(self, session_id=None):
        return None

    def set_fold(self, fold_index):
        self._fold = int(fold_index)

    def train_numpy(self):
        if self._kfold:
            # Leave-one-chunk-out style folds over contiguous blocks.
            n = len(self._y)
            fold_size = max(1, n // self.n_folds)
            start = self._fold * fold_size
            end = n if self._fold == self.n_folds - 1 else start + fold_size
            mask = np.ones(n, dtype=bool)
            mask[start:end] = False
            return self._X[mask], self._y[mask]
        n = int(0.7 * len(self._y))
        return self._X[:n], self._y[:n]

    def test_numpy(self):
        if self._kfold:
            n = len(self._y)
            fold_size = max(1, n // self.n_folds)
            start = self._fold * fold_size
            end = n if self._fold == self.n_folds - 1 else start + fold_size
            return self._X[start:end], self._y[start:end]
        n = int(0.7 * len(self._y))
        return self._X[n:], self._y[n:]

    def get_data(self):
        import pandas as pd

        df = pd.DataFrame(self._X, columns=[f"f{i}" for i in range(self._X.shape[1])])
        return df, pd.Series(self._y, name="target")

    def split_description(self):
        return "tiny synthetic"


def _unfis(D=4, rules=2, out_features=2, binary=True, **kwargs):
    return UNFIS(D, rules=rules, out_features=out_features, binary=binary, **kwargs)


def _eq8_numpy(mu, zeta, eps):
    return (mu + eps) / ((1.0 - zeta) * mu + zeta + eps)


# ---------------------------------------------------------------------------
# Architecture / equations
# ---------------------------------------------------------------------------


def test_selector_sigmoid_range():
    m = _unfis()
    with torch.no_grad():
        m.selector_logits.normal_(0.0, 1.0)
    zeta = m.selection_strengths()
    assert zeta.shape == (m.rules_count, m.in_features)
    assert torch.all(zeta > 0.0) and torch.all(zeta < 1.0)


def test_selected_membership_limits():
    m = _unfis(D=3, rules=1)
    x = torch.randn(5, 3)
    mu = m.membership(x)
    with torch.no_grad():
        m.selector_logits.fill_(20.0)
    sel_hi = m.selected_memberships(x)
    assert torch.allclose(sel_hi, mu, atol=1e-4)

    with torch.no_grad():
        m.selector_logits.fill_(-20.0)
    sel_lo = m.selected_memberships(x)
    assert torch.allclose(sel_lo, torch.ones_like(sel_lo), atol=1e-4)


def test_selected_membership_exact_equation():
    m = _unfis(D=3, rules=2, selection_epsilon=1e-6)
    x = torch.randn(4, 3)
    with torch.no_grad():
        m.selector_logits.uniform_(-2.0, 2.0)
    mu = m.membership(x).detach().numpy()
    zeta = m.selection_strengths().detach().numpy()
    got = m.selected_memberships(x).detach().numpy()
    ref = _eq8_numpy(mu, zeta[None, ...], m.selection_epsilon)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_gaussian_membership_at_center():
    m = _unfis(D=2, rules=2)
    with torch.no_grad():
        m.centers.copy_(torch.tensor([[0.0, 1.0], [2.0, -1.0]]))
    x = m.centers.detach().clone()
    mu = m.membership(x)
    # Diagonal samples at own rule centers → μ=1 for matching rule/feature.
    assert torch.allclose(mu[0, 0], torch.ones(2), atol=1e-6)
    assert torch.allclose(mu[1, 1], torch.ones(2), atol=1e-6)


def test_selected_consequent_uses_same_zeta():
    m = _unfis(D=2, rules=1, out_features=2)
    with torch.no_grad():
        m.selector_logits[0, 0] = 20.0  # active
        m.selector_logits[0, 1] = -20.0  # relaxed
        m.consequent_weights.fill_(1.0)
        m.consequent_bias.zero_()
        m.class_thresholds.zero_()
        # Force single rule weight 1 via large separation
        m.centers.fill_(0.0)
        m.raw_spreads.fill_(0.0)
    x = torch.tensor([[3.0, 5.0]])
    local = m.local_rule_outputs(x)  # [1,1,2]
    # Feature 1 relaxed ⇒ only x0 contributes ≈ 3
    assert torch.allclose(local[0, 0], torch.tensor([3.0, 3.0]), atol=1e-3)
    sel = m.selected_memberships(x)
    # ζ→0 ⇒ selected_mu → 1 (Eq. 8); ε keeps it slightly below 1.
    assert float(sel[0, 0, 1]) > 0.99


def test_log_product_matches_direct_product():
    m = _unfis(D=3, rules=2)
    x = torch.randn(6, 3)
    sel = m.selected_memberships(x).clamp_min(1e-6)
    direct = sel.prod(dim=2)
    via_log = torch.exp(m.log_firing_strengths(x))
    assert torch.allclose(direct, via_log, rtol=1e-4, atol=1e-5)


def test_normalized_firing_sums_to_one():
    m = _unfis()
    phi = m.normalized_firing_strengths(torch.randn(10, 4))
    assert phi.shape == (10, 2)
    assert torch.allclose(phi.sum(dim=1), torch.ones(10), atol=1e-5)


def test_class_score_shape():
    m_bin = _unfis(binary=True, out_features=2)
    assert m_bin.out_features == 2
    scores = m_bin.class_scores(torch.randn(5, 4))
    assert scores.shape == (5, 2)

    m_mc = UNFIS(3, rules=2, out_features=4, binary=False)
    scores_mc = m_mc.class_scores(torch.randn(5, 3))
    assert scores_mc.shape == (5, 4)


def test_probabilities_are_valid():
    m = _unfis()
    p = m.probabilities(torch.randn(8, 4))
    assert torch.isfinite(p).all()
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0)
    assert torch.allclose(p.sum(dim=1), torch.ones(8), atol=1e-5)


def test_original_label_roundtrip():
    # Non-contiguous ints
    m = _unfis()
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.array([2 if v > 0 else 5 for v in X[:, 0]])
    w = SklearnUNFISWrapper(m, batch_size=8)
    w.fit(X, y, max_iterations=2, minibatch_size=8, random_state=0)
    pred = w.predict(X)
    assert set(np.unique(pred)).issubset({2, 5})

    # String labels
    m2 = _unfis()
    y_s = np.array(["negative" if v <= 0 else "positive" for v in X[:, 0]])
    w2 = SklearnUNFISWrapper(m2, batch_size=8)
    w2.fit(X, y_s, max_iterations=2, minibatch_size=8, random_state=1)
    pred_s = w2.predict(X[:5])
    assert all(p in ("negative", "positive") for p in pred_s.tolist())


def test_positive_spreads():
    m = _unfis()
    X = np.random.randn(30, 4)
    y = (X[:, 0] > 0).astype(int)
    knn_initialize_unfis(m, X, y, random_state=0)
    assert torch.all(m.spreads > 0)
    gqlm_train_unfis(m, X, y, max_iterations=3, minibatch_size=8, random_state=0)
    assert torch.all(m.spreads > 0)


# ---------------------------------------------------------------------------
# KNN init
# ---------------------------------------------------------------------------


def test_knn_initialization_is_deterministic():
    X = np.random.RandomState(0).randn(25, 3)
    y = (X[:, 0] > 0).astype(int)
    a = _unfis(D=3)
    b = _unfis(D=3)
    knn_initialize_unfis(a, X, y, random_state=7, selector_init_std=0.01)
    knn_initialize_unfis(b, X, y, random_state=7, selector_init_std=0.01)
    assert torch.allclose(a.centers, b.centers)
    assert torch.allclose(a.raw_spreads, b.raw_spreads)
    assert torch.allclose(a.selector_logits, b.selector_logits)
    assert torch.allclose(a.consequent_weights, b.consequent_weights)


def test_knn_initialization_uses_training_data_only():
    """Initializer must not see held-out rows (spy on NearestNeighbors.fit)."""
    from sklearn.neighbors import NearestNeighbors

    X_train = np.random.RandomState(1).randn(20, 3)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = np.random.RandomState(2).randn(8, 3)

    seen = []

    class SpyNN(NearestNeighbors):
        def fit(self, X, y=None):
            seen.append(np.asarray(X).shape[0])
            return super().fit(X, y)

    m = _unfis(D=3)
    with mock.patch("model.unfis.NearestNeighbors", SpyNN):
        knn_initialize_unfis(m, X_train, y_train, random_state=0)
    assert seen, "Expected NearestNeighbors.fit during KNN init"
    assert all(n <= len(X_train) for n in seen)
    assert max(seen) <= len(X_train)
    # Sanity: test fold size never appears as a larger fit than train.
    assert max(seen) != len(X_train) + len(X_test)


def test_singleton_cluster_width_guard():
    # Force tiny data so clusters can be singletons.
    X = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    y = np.array([0, 1])
    m = _unfis(D=2, rules=2)
    knn_initialize_unfis(m, X, y, random_state=0)
    assert torch.all(m.spreads >= m.sigma_min - 1e-12)


# ---------------------------------------------------------------------------
# GqLM
# ---------------------------------------------------------------------------


def test_gqlm_parameters_change():
    m = _unfis()
    X = np.random.RandomState(0).randn(24, 4)
    y = (X[:, 0] > 0).astype(int)
    knn_initialize_unfis(m, X, y, random_state=0)
    before = m.pack_parameters().detach().clone()
    # eta is a denominator (Eq. 30); eta=1 ⇒ unscaled J+ step.
    gqlm_train_unfis(
        m, X, y, max_iterations=5, minibatch_size=8, random_state=0, eta=1.0
    )
    after = m.pack_parameters().detach()
    assert not torch.allclose(before, after)


def test_gqlm_cross_entropy_reduces_on_tiny_dataset():
    rng = np.random.RandomState(0)
    X = rng.randn(16, 2)
    # Easily separable
    X[:8, 0] -= 2.0
    X[8:, 0] += 2.0
    y = np.array([0] * 8 + [1] * 8)
    m = _unfis(D=2, rules=2)
    train_unfis_model(
        m,
        X,
        y,
        max_iterations=30,
        minibatch_size=16,
        random_state=0,
        # Paper Table 2: η=1e−3 as Eq. (30) denominator; λ=1e3 damping.
        eta=1e-3,
        lambda_=1e3,
        beta=0.9,
    )
    hist = m.training_metadata["loss_history"]
    assert hist[-1] < hist[0]
    assert m.training_metadata["gqlm_equation_30_scale"] == "1/eta"
    assert m.training_metadata["eta_source"] == "paper"


def test_gqlm_no_adam_or_onecycle():
    experiment = _TinyExperiment(n_samples=30, n_features=3, seed=0)
    device = torch.device("cpu")
    configs = {
        "UNFIS": {
            "model_class": UNFIS,
            "wrapper_class": SklearnUNFISWrapper,
            "params": {
                "rules": 2,
                "out_features": 2,
                "binary": True,
                "drop_out_p": 0.0,
            },
            "training": {
                "trainer": "custom_fit",
                "fit_method": "unfis_gqlm",
                "uses_reconstruction": False,
                "scheduler": "none",
                "fit_params": {
                    "minibatch_size": 8,
                    "max_iterations": 2,
                    "selector_init_std": 0.01,
                },
            },
        }
    }
    learning_params = {
        "batch_size": 8,
        "lr": 0.01,
        "max_lr": 0.01,
        "epochs": 2,
        "alpha": 1.0,
        "min_alpha": 0.0,
    }
    adam_calls = {"n": 0}
    oc_calls = {"n": 0}

    real_adam = torch.optim.Adam

    def counting_adam(*a, **k):
        adam_calls["n"] += 1
        return real_adam(*a, **k)

    def counting_onecycle(*a, **k):
        oc_calls["n"] += 1
        raise AssertionError("OneCycleLR should not be constructed on custom_fit path")

    with mock.patch("tests.evaluate.torch.optim.Adam", side_effect=counting_adam), mock.patch(
        "tests.evaluate.OneCycleLR", side_effect=counting_onecycle
    ):
        ev = Evaluator(
            experiment=experiment,
            model_configs=configs,
            learning_params=learning_params,
            device=device,
            binary=True,
            noise_levels=[0.0],
            n_runs=1,
            random_state=1,
            use_noise=False,
            use_early_stopping=False,
        )
        results = ev.evaluate(verbose=False)
    assert adam_calls["n"] == 0
    assert oc_calls["n"] == 0
    assert len(results) == 1
    assert "UNFIS" in ev._last_wrapper
    assert isinstance(ev._last_wrapper["UNFIS"], SklearnUNFISWrapper)


def test_gqlm_jacobian_finite():
    m = _unfis(D=2, rules=1)
    X = torch.randn(4, 2, requires_grad=False)
    y = np.array([0, 1, 0, 1])
    knn_initialize_unfis(m, X.numpy(), y, random_state=0)
    J, probs = _build_probability_jacobian(m, X)
    assert torch.isfinite(J).all()
    assert torch.isfinite(probs).all()
    assert J.shape[0] == 4 * 2


def test_analytic_jacobian_matches_autograd():
    m = _unfis(D=2, rules=1)
    X = np.random.randn(6, 2)
    y = (X[:, 0] > 0).astype(int)
    knn_initialize_unfis(m, X, y, random_state=0)
    with pytest.raises(NotImplementedError):
        gqlm_train_unfis(
            m, X, y, max_iterations=1, jacobian_mode="analytic", random_state=0
        )


def test_linear_solver_fallback_is_finite():
    J = torch.randn(6, 4, dtype=torch.float64)
    # Force near-singular by making columns dependent
    J[:, 3] = J[:, 0]
    rhs = torch.randn(4, dtype=torch.float64)
    with mock.patch("torch.linalg.cholesky", side_effect=RuntimeError("boom")):
        delta, solver = _solve_damped_normal(
            J, residual=None, lambda_=1e-8, rhs=rhs, n_classes=2
        )
    assert solver in {"solve", "lstsq"}
    assert torch.isfinite(delta).all()


def _class_wise_probability_jacobians(model: UNFIS, X: torch.Tensor):
    """Return list of J_c with shape [B, P] for c = 0..C-1."""
    params = model.trainable_parameters()
    probs = model.probabilities(X)
    B, C = probs.shape
    jcs = []
    for c in range(C):
        rows = []
        for b in range(B):
            grads = torch.autograd.grad(
                probs[b, c],
                params,
                retain_graph=True,
                allow_unused=True,
            )
            parts = []
            for p, g in zip(params, grads):
                if g is None:
                    parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
                else:
                    parts.append(g.reshape(-1))
            rows.append(torch.cat(parts))
        jcs.append(torch.stack(rows, dim=0))
    return jcs, probs.detach()


def test_stacked_jacobian_damping_matches_class_wise():
    """Eq. (31): Σ_c (J_c^T J_c + λ I) == J_stacked^T J_stacked + C λ I."""
    torch.manual_seed(0)
    m = UNFIS(2, rules=1, out_features=2, binary=True, dtype=torch.float64)
    X = torch.randn(5, 2, dtype=torch.float64)
    y = np.array([0, 1, 0, 1, 0])
    knn_initialize_unfis(m, X.numpy(), y, random_state=0)
    # Keep float64 after init
    m = m.to(dtype=torch.float64)

    lambda_ = 1.5
    C = m.out_features
    jcs, _ = _class_wise_probability_jacobians(m, X)
    P = jcs[0].shape[1]
    I = torch.eye(P, dtype=torch.float64)
    H_ref = torch.zeros(P, P, dtype=torch.float64)
    for Jc in jcs:
        H_ref = H_ref + Jc.T @ Jc + float(lambda_) * I

    J_stacked = torch.cat(jcs, dim=0)
    H_stacked = J_stacked.T @ J_stacked + float(C) * float(lambda_) * I
    assert torch.allclose(H_ref, H_stacked, rtol=1e-10, atol=1e-10)

    rhs = torch.randn(P, dtype=torch.float64)
    d_ref = torch.linalg.solve(H_ref, rhs)
    d_impl, _ = _solve_damped_normal(
        J_stacked,
        residual=None,
        lambda_=lambda_,
        rhs=rhs,
        n_classes=C,
        solver_dtype="float64",
    )
    assert torch.allclose(d_ref, d_impl, rtol=1e-10, atol=1e-10)


def test_eta_scales_as_one_over_eta():
    torch.manual_seed(1)
    J = torch.randn(8, 5, dtype=torch.float64)
    rhs = torch.randn(5, dtype=torch.float64)
    base, _ = _solve_damped_normal(
        J, residual=None, lambda_=2.0, rhs=rhs, n_classes=2, solver_dtype="float64"
    )
    # Eq. (30): Δπ = (1/η) * J+ * rhs_vec
    for eta in (2.0, 0.5):
        expected = base / float(eta)
        wrong_old = base * float(eta)
        got = base / float(eta)
        assert torch.allclose(got, expected)
        assert not torch.allclose(got, wrong_old)

    with pytest.raises(ValueError, match="eta must be > 0"):
        gqlm_train_unfis(
            _unfis(D=2),
            np.random.randn(8, 2),
            np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            eta=0.0,
            max_iterations=1,
        )
    with pytest.raises(ValueError, match="eta must be > 0"):
        gqlm_train_unfis(
            _unfis(D=2),
            np.random.randn(8, 2),
            np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            eta=-1.0,
            max_iterations=1,
        )


def test_gqlm_update_matches_direct_eq30_eq31():
    """One β=0 step must match direct class-wise Eqs. (30)–(31)."""
    torch.manual_seed(2)
    np.random.seed(2)
    m = UNFIS(2, rules=1, out_features=2, binary=True, dtype=torch.float64)
    X = np.array([[0.2, -0.1], [-0.3, 0.4], [0.5, 0.1], [-0.2, -0.5]], dtype=np.float64)
    y = np.array([0, 1, 0, 1])
    knn_initialize_unfis(m, X, y, random_state=0, selector_init_std=0.01)
    m = m.to(dtype=torch.float64)

    lambda_ = 10.0
    eta = 2.0
    X_t = torch.as_tensor(X, dtype=torch.float64)
    Y_oh = torch.as_tensor(np.eye(2)[y], dtype=torch.float64)

    before = m.pack_parameters().detach().clone()
    jcs, probs = _class_wise_probability_jacobians(m, X_t)
    C = len(jcs)
    P = jcs[0].shape[1]
    I = torch.eye(P, dtype=torch.float64)
    H = torch.zeros(P, P, dtype=torch.float64)
    rhs = torch.zeros(P, dtype=torch.float64)
    for c, Jc in enumerate(jcs):
        H = H + Jc.T @ Jc + float(lambda_) * I
        xi_p = (Y_oh[:, c] / probs[:, c].clamp_min(1e-12)).to(dtype=torch.float64)
        rhs = rhs + Jc.T @ xi_p
    delta_ref = (1.0 / float(eta)) * torch.linalg.solve(H, rhs)

    gqlm_train_unfis(
        m,
        X,
        y,
        minibatch_size=len(X),
        max_iterations=1,
        beta=0.0,
        eta=eta,
        lambda_=lambda_,
        random_state=0,
        solver_dtype="float64",
    )
    after = m.pack_parameters().detach()
    delta_impl = after - before
    assert torch.allclose(delta_impl, delta_ref, rtol=1e-8, atol=1e-8)
    assert m.training_metadata["gqlm_equation_31_damping"] == (
        "C*lambda_for_stacked_jacobian"
    )
    assert m.training_metadata["solver_dtype"] == "float64"


def test_solver_dtype_is_honored():
    m = _unfis(D=2)
    X = np.random.RandomState(0).randn(12, 2)
    y = (X[:, 0] > 0).astype(int)
    train_unfis_model(
        m, X, y, max_iterations=1, solver_dtype="float32", random_state=0, eta=1.0
    )
    assert m.training_metadata["solver_dtype"] == "float32"
    with pytest.raises(ValueError, match="Unsupported solver_dtype"):
        gqlm_train_unfis(m, X, y, max_iterations=1, solver_dtype="float16")


def test_wrapper_uses_model_parameter_dtype():
    m = UNFIS(3, rules=2, out_features=2, binary=True, dtype=torch.float64)
    X = np.random.RandomState(0).randn(10, 3).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)
    w = SklearnUNFISWrapper(m, dtype=torch.float32, batch_size=4)
    # Leave model in float64; wrapper self.dtype stays float32
    w.model = m.to(dtype=torch.float64)
    knn_initialize_unfis(w.model, X, y, random_state=0)
    w.model.is_fitted_ = True
    w.classes_ = np.asarray(w.model._classes)
    w.is_fitted_ = True
    proba = w.predict_proba(X)
    assert proba.shape == (10, 2)
    assert np.isfinite(proba).all()


def test_gqlm_hyperparameter_validation():
    m = _unfis(D=2)
    X = np.random.randn(8, 2)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="minibatch_size"):
        gqlm_train_unfis(m, X, y, minibatch_size=0, max_iterations=1)
    with pytest.raises(ValueError, match="lambda_"):
        gqlm_train_unfis(m, X, y, lambda_=-1.0, max_iterations=1)
    with pytest.raises(ValueError, match="beta"):
        gqlm_train_unfis(m, X, y, beta=1.0, max_iterations=1)
    with pytest.raises(ValueError, match="max_iterations"):
        gqlm_train_unfis(m, X, y, max_iterations=0)


def test_relaxation_rate_matches_one_minus_zeta():
    m = _unfis()
    with torch.no_grad():
        m.selector_logits.uniform_(-1.0, 1.0)
    zeta = m.selection_strengths()
    overall, per = m.relaxation_rate(per_rule=True)
    assert np.isclose(overall, float((1.0 - zeta).mean()))
    assert torch.allclose(per, (1.0 - zeta).mean(dim=1))


def test_linguistic_richness_is_structural_zero():
    m = _unfis(rules=3)
    assert m.linguistic_richness() == 0.0
    overall, per = m.linguistic_richness(per_rule=True)
    assert overall == 0.0
    assert torch.equal(per, torch.zeros(3, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Wrapper / checkpoints
# ---------------------------------------------------------------------------


def test_wrapper_batched_inference():
    m = _unfis()
    X = np.random.randn(20, 4).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    w = SklearnUNFISWrapper(m, batch_size=7)
    w.fit(X, y, max_iterations=1, minibatch_size=10, random_state=0)

    sizes = []
    orig = m.class_scores

    def spy(x):
        sizes.append(int(x.shape[0]))
        return orig(x)

    m.class_scores = spy  # type: ignore[method-assign]
    _ = w.predict_proba(X)
    assert sizes
    assert max(sizes) <= 7


def test_wrapper_empty_input():
    m = _unfis()
    X = np.random.randn(10, 4).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    w = SklearnUNFISWrapper(m)
    w.fit(X, y, max_iterations=1, random_state=0)
    empty = np.empty((0, 4), dtype=np.float32)
    assert w.predict(empty).shape == (0,)
    assert w.predict_proba(empty).shape == (0, 2)
    assert w.decision_function(empty).shape == (0, 2)


def test_checkpoint_roundtrip():
    m = _unfis()
    X = np.random.randn(16, 4)
    y = (X[:, 0] > 0).astype(int)
    train_unfis_model(m, X, y, max_iterations=2, random_state=0)
    sd = m.state_dict()
    assert "centers" in sd and "mean" not in sd
    m2 = _unfis()
    m2.load_state_dict(sd, strict=True)
    x = torch.randn(3, 4)
    assert torch.allclose(m.class_scores(x), m2.class_scores(x), atol=1e-5)
    assert m2.is_fitted_
    assert m2._classes is not None


def test_legacy_checkpoint_aliases():
    m = _unfis()
    sd = m.state_dict()
    legacy = {
        "mean": sd["centers"].T.contiguous(),  # old [D, R]
        "raw_std": sd["raw_spreads"].T.contiguous(),
        "s": sd["selector_logits"].T.contiguous(),
        "consequent_weight": sd["consequent_weights"].clone(),
        "consequent_bias": sd["consequent_bias"].clone(),
        "theta": sd["class_thresholds"].clone(),
        "decoder_linear.weight": sd["decoder_linear.weight"],
        "decoder_linear.bias": sd["decoder_linear.bias"],
        "_extra_state": sd["_extra_state"],
    }
    loaded = _unfis()
    incompatible = loaded.load_state_dict(legacy, strict=True)
    assert incompatible.missing_keys == []
    assert torch.allclose(loaded.centers, sd["centers"])


# ---------------------------------------------------------------------------
# Evaluator integration
# ---------------------------------------------------------------------------


def test_evaluator_uses_custom_fit_path():
    experiment = _TinyExperiment(n_samples=28, n_features=3, seed=3)
    configs = {
        "UNFIS": {
            "model_class": UNFIS,
            "wrapper_class": SklearnUNFISWrapper,
            "params": {"rules": 2, "out_features": 2, "binary": True, "drop_out_p": 0.0},
            "training": {
                "trainer": "custom_fit",
                "fit_method": "unfis_gqlm",
                "uses_reconstruction": False,
                "scheduler": "none",
                "fit_params": {"max_iterations": 2, "minibatch_size": 8},
            },
        }
    }
    learning_params = {
        "batch_size": 8,
        "lr": 0.01,
        "max_lr": 0.01,
        "epochs": 2,
        "alpha": 1.0,
        "min_alpha": 0.0,
    }
    ev = Evaluator(
        experiment=experiment,
        model_configs=configs,
        learning_params=learning_params,
        device=torch.device("cpu"),
        binary=True,
        noise_levels=[0.0],
        n_runs=1,
        random_state=2,
        use_noise=False,
        use_early_stopping=False,
    )
    results = ev.evaluate(verbose=False)
    assert len(results) == 1
    assert np.isfinite(results["test_acc"].iloc[0])
    assert "UNFIS" in ev._last_wrapper
    assert isinstance(ev._last_wrapper["UNFIS"], SklearnUNFISWrapper)


def test_existing_models_keep_default_evaluator_path():
    experiment = _TinyExperiment(n_features=4)
    configs = {
        "Anfis": {
            "model_class": ANFIS,
            "wrapper_class": SklearnAnfisWrapper,
            "params": {"rules": 2, "drop_out_p": 0.0},
        }
    }
    learning_params = {
        "batch_size": 8,
        "lr": 0.01,
        "max_lr": 0.01,
        "epochs": 2,
        "alpha": 0.5,
        "min_alpha": 0.1,
    }
    adam_calls = {"n": 0}
    real_adam = torch.optim.Adam

    def counting_adam(*a, **k):
        adam_calls["n"] += 1
        return real_adam(*a, **k)

    with mock.patch("tests.evaluate.torch.optim.Adam", side_effect=counting_adam):
        ev = Evaluator(
            experiment=experiment,
            model_configs=configs,
            learning_params=learning_params,
            device=torch.device("cpu"),
            binary=True,
            noise_levels=[0.0],
            n_runs=1,
            random_state=1,
            use_noise=False,
            use_early_stopping=False,
        )
        results = ev.evaluate(verbose=False)
    assert adam_calls["n"] >= 1
    assert len(results) == 1


def test_admtsk_path_is_unchanged():
    experiment = _TinyExperiment(n_samples=36, n_features=4, seed=4)
    configs = {
        "ADMTSK": {
            "model_class": ADMTSK,
            "wrapper_class": SklearnADMTSKWrapper,
            "params": {
                "rules": 3,
                "out_features": 2,
                "binary": True,
                "paper_mode": True,
            },
            "training": {
                "loss_mode": "paper_mse",
                "uses_reconstruction": False,
                "scheduler": "none",
            },
        }
    }
    learning_params = {
        "batch_size": 8,
        "lr": 0.01,
        "max_lr": 0.01,
        "epochs": 2,
        "alpha": 0.0,
        "min_alpha": 0.0,
    }
    ev = Evaluator(
        experiment=experiment,
        model_configs=configs,
        learning_params=learning_params,
        device=torch.device("cpu"),
        binary=True,
        noise_levels=[0.0],
        n_runs=1,
        random_state=5,
        use_noise=False,
        use_early_stopping=False,
    )
    results = ev.evaluate(verbose=False)
    assert len(results) == 1
    assert np.isfinite(results["test_acc"].iloc[0])


def test_ten_fold_smoke():
    experiment = _TinyExperiment(
        n_samples=50, n_features=3, seed=9, n_folds=10, kfold=True
    )
    configs = {
        "UNFIS": {
            "model_class": UNFIS,
            "wrapper_class": SklearnUNFISWrapper,
            "params": {"rules": 2, "out_features": 2, "binary": True, "drop_out_p": 0.0},
            "training": {
                "trainer": "custom_fit",
                "fit_method": "unfis_gqlm",
                "uses_reconstruction": False,
                "scheduler": "none",
                "fit_params": {"max_iterations": 1, "minibatch_size": 16},
            },
        }
    }
    learning_params = {
        "batch_size": 8,
        "lr": 0.01,
        "max_lr": 0.01,
        "epochs": 1,
        "alpha": 0.0,
        "min_alpha": 0.0,
    }
    ev = Evaluator(
        experiment=experiment,
        model_configs=configs,
        learning_params=learning_params,
        device=torch.device("cpu"),
        binary=True,
        noise_levels=[0.0],
        n_runs=10,
        random_state=11,
        use_noise=False,
        use_early_stopping=False,
    )
    results = ev.evaluate(verbose=False)
    assert len(results) == 10
    assert results["test_acc"].notna().all()
