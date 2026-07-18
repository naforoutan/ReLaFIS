"""Focused tests for paper-aligned VSRP-AnYa-EFS."""

from __future__ import annotations

import copy
import math
from unittest import mock

import numpy as np
import pytest
import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from model.vsrp_anya import (
    PROJECTION_FIXED_PER_CLOUD,
    PROJECTION_PAPER_DYNAMIC,
    SklearnVSRPAnyaEFSWrapper,
    VSRPAnyaEFS,
    local_wrls_step_numpy,
    make_rsb_matrix_numpy,
    projected_dimension,
)


def _toy_binary(n=40, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _toy_multi(n=60, d=4, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float64)
    y = rng.integers(0, 3, size=n)
    return X, y


# ---------------------------------------------------------------------------
# RSB / projection
# ---------------------------------------------------------------------------
def test_rsb_alphabet_and_scale():
    d_ext, L = 9, 4
    s = math.sqrt(d_ext)
    mag = math.sqrt(s)
    R = make_rsb_matrix_numpy(d_ext, L, np.random.default_rng(0))
    flat = R.reshape(-1)
    allowed = np.array([-mag, 0.0, mag])
    assert np.all(np.any(np.isclose(flat[:, None], allowed[None, :], atol=1e-12), axis=1))


def test_rsb_empirical_probabilities():
    d_ext = 16
    s = math.sqrt(d_ext)
    mag = math.sqrt(s)
    p = 1.0 / (2.0 * s)
    R = make_rsb_matrix_numpy(d_ext, 20000, np.random.default_rng(123))
    flat = R.reshape(-1)
    frac_pos = np.mean(np.isclose(flat, mag))
    frac_neg = np.mean(np.isclose(flat, -mag))
    frac_zero = np.mean(np.isclose(flat, 0.0))
    assert abs(frac_pos - p) < 0.01
    assert abs(frac_neg - p) < 0.01
    assert abs(frac_zero - (1.0 - 1.0 / s)) < 0.01


def test_projection_includes_1_over_sqrt_L():
    m = VSRPAnyaEFS(3, 1, 2, binary=True, compression_ratio=2, seed=0, dtype=torch.float32)
    X = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]], dtype=np.float64)
    y = np.array([0, 1])
    m.fit(X, y)
    R = m._rsb_for_train_step(0)
    xe = np.concatenate([[1.0], X[1]])  # second sample is first online step
    U = m._project_u_shared(X[1], R)
    expected = (1.0 / math.sqrt(m.proj_dim)) * (R.T @ xe)
    assert np.allclose(U[0], expected)


def test_projected_dimension_calculation():
    assert projected_dimension(11, 3) == math.ceil(11 / 3)
    assert projected_dimension(5, 2) == 3


def test_reject_proj_dim_greater_than_d_ext():
    with pytest.raises(ValueError, match="cannot exceed"):
        VSRPAnyaEFS(3, 1, 2, binary=True, proj_dim=10, dtype=torch.float32)


# ---------------------------------------------------------------------------
# First-sample Algorithm 1 init
# ---------------------------------------------------------------------------
def test_first_sample_initializes_without_wrls():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, p_init=500.0, dtype=torch.float32)
    x = np.array([[0.25, -0.5]])
    m.partial_fit(x, np.array([0]), classes=[0, 1])
    assert m.rules_count == 1
    assert float(m.radii[0]) == pytest.approx(1.0)
    assert torch.allclose(m.consequent, torch.zeros_like(m.consequent))
    assert np.allclose(m._P[0], 500.0 * np.eye(m.proj_dim))
    assert float(m.supports[0]) == pytest.approx(1.0)
    assert np.allclose(m.centers[0].detach().cpu().numpy(), x[0])
    assert np.allclose(m.foci[0].detach().cpu().numpy(), x[0])
    assert float(m.avg_sq[0]) == pytest.approx(float(np.dot(x[0], x[0])))
    assert m._n_seen == 1
    assert m._train_step_index == 0


def test_second_sample_updates_q_and_covariance():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, p_init=500.0, dtype=torch.float32)
    m.partial_fit(np.array([[0.0, 0.0]]), np.array([0]), classes=[0, 1])
    Q0 = m.consequent.detach().cpu().numpy().copy()
    P0 = m._P[0].copy()
    m.partial_fit(np.array([[0.3, -0.2]]), np.array([1]))
    assert m._train_step_index == 1
    assert m._n_seen == 2
    assert not np.allclose(m.consequent.detach().cpu().numpy(), Q0) or not np.allclose(
        m._P[0], P0
    )


def test_new_cloud_radius_stays_one_until_updated():
    m = VSRPAnyaEFS(1, 1, 2, binary=True, seed=0, p_init=500.0, dtype=torch.float32)
    X = np.array([[0.0], [10.0], [20.0]], dtype=np.float64)
    y = np.array([0, 1, 0])
    m.fit(X, y)
    for i in range(m.rules_count):
        if not m._cloud_updated_flags[i]:
            assert float(m.radii[i]) == pytest.approx(1.0)


def test_radius_recursion_once_on_update():
    m = VSRPAnyaEFS(1, 1, 2, binary=True, seed=0, dtype=torch.float32, max_rules=1)
    m.fit(np.array([[0.0], [0.1]]), np.array([0, 1]))
    assert m.rules_count == 1
    assert m._cloud_updated_flags[0] is True
    assert float(m.radii[0]) != pytest.approx(1.0)


def test_global_and_local_recursions():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32, max_rules=1)
    x1 = np.array([1.0, 2.0])
    x2 = np.array([3.0, 4.0])
    m.fit(np.vstack([x1, x2]), np.array([0, 1]))
    assert np.allclose(m._global_mean, 0.5 * (x1 + x2))
    assert m._global_avg_sq == pytest.approx(
        0.5 * (np.dot(x1, x1) + np.dot(x2, x2))
    )
    g = m.centers[0].detach().cpu().numpy()
    assert np.allclose(g, 0.5 * (x1 + x2))


def test_global_and_local_density_manual():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0, 1]))
    v = np.array([0.5, 0.0])
    dens = m._global_density(v)
    gm = m._global_mean
    dist2 = float(np.dot(v - gm, v - gm))
    sigma2 = max(m._global_avg_sq - float(np.dot(gm, gm)), m.density_epsilon)
    assert dens == pytest.approx(1.0 / (1.0 + dist2 / sigma2))
    i = 0
    ld = m._local_density_np(v, i)
    g = m.centers[i].detach().cpu().numpy()
    dist2 = float(np.dot(v - g, v - g))
    sigma2 = max(float(m.avg_sq[i]) - float(np.dot(g, g)), m.density_epsilon)
    assert ld == pytest.approx(1.0 / (1.0 + dist2 / sigma2))


def test_normalized_densities_sum_to_one():
    m = VSRPAnyaEFS(3, 1, 2, binary=True, seed=0, dtype=torch.float32)
    X, y = _toy_binary(30, 3)
    m.fit(X, y)
    lam = m._firing_np(X[0])
    assert np.isclose(lam.sum(), 1.0)
    assert np.all(lam >= -1e-12)


def test_density_extreme_and_distance_criteria():
    m = VSRPAnyaEFS(1, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m._classes = np.array([0, 1])
    m._reset_state()
    m._classes = np.array([0, 1])
    m._bootstrap_first_cloud(np.array([0.0]))
    assert m._should_add_cloud(np.array([100.0])) is True


def test_nearest_cloud_update_keeps_focal_mean_distinct():
    m = VSRPAnyaEFS(1, 1, 2, binary=True, seed=0, max_rules=1, dtype=torch.float32)
    m.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    assert m.rules_count == 1
    focus = m.foci[0].detach().cpu().numpy()
    mean = m.centers[0].detach().cpu().numpy()
    assert np.allclose(focus, np.array([1.0]))
    assert not np.allclose(focus, mean)


def test_rules_hint_does_not_cap_evolution():
    m = VSRPAnyaEFS(1, rules=1, out_features=2, binary=True, seed=0, dtype=torch.float32)
    X = np.linspace(-5, 5, 15).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)
    m.fit(X, y)
    assert m.rules_hint == 1
    assert m.max_rules is None
    assert m.rules_count > 1


def test_explicit_max_rules_cap_recorded():
    m = VSRPAnyaEFS(1, 1, 2, binary=True, seed=0, max_rules=2, dtype=torch.float32)
    X = np.linspace(-10, 10, 20).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)
    m.fit(X, y)
    assert m.rules_count <= 2
    assert any("max_rules" in a for a in m.model_metadata["adaptations"])


# ---------------------------------------------------------------------------
# Classification schema
# ---------------------------------------------------------------------------
def test_binary_two_outputs_one_hot_argmax():
    m = VSRPAnyaEFS(3, 1, 1, binary=True, seed=0, dtype=torch.float32)
    assert m.out_features == 2
    X, y = _toy_binary(25, 3)
    w = SklearnVSRPAnyaEFSWrapper(m)
    w.fit(X, y)
    assert w.predict_proba(X).shape[1] == 2


def test_multiclass_stable_order_and_reject_unseen():
    X, y = _toy_multi(40, 4)
    m = VSRPAnyaEFS(4, 1, 3, binary=False, seed=0, dtype=torch.float32)
    m.partial_fit(X[:10], y[:10], classes=[0, 1, 2])
    assert list(m._classes) == [0, 1, 2]
    with pytest.raises(ValueError, match="Unseen class"):
        m.partial_fit(X[:2], np.array([0, 9]))


def test_partial_fit_requires_classes_first():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="classes="):
        m.partial_fit(np.array([[0.0, 0.0]]), np.array([0]))


# ---------------------------------------------------------------------------
# Antecedents / output / wRLS
# ---------------------------------------------------------------------------
def test_compressed_antecedent_and_system_output():
    m = VSRPAnyaEFS(
        2, 1, 2, binary=True, seed=0, compression_ratio=2, dtype=torch.float32
    )
    X = np.array([[0.2, -0.3], [0.1, 0.4]], dtype=np.float64)
    y = np.array([0, 1])
    m.fit(X, y)
    x = X[0]
    lam = m._firing_np(x)
    R = m._get_inference_R()
    U = m._project_u_shared(x, R)
    Q = m.consequent.detach().cpu().numpy()
    manual = np.zeros(2)
    for i in range(m.rules_count):
        manual += lam[i] * (U[i] @ Q[i])
    scores, _ = m(torch.as_tensor(x[None, :], dtype=torch.float32))
    assert np.allclose(scores.detach().cpu().numpy()[0], manual, atol=1e-5)


def test_local_wrls_matches_numpy_reference():
    rng = np.random.default_rng(0)
    L, C = 5, 2
    Cmat = rng.normal(size=(L, L))
    Cmat = Cmat @ Cmat.T + np.eye(L)
    Q = rng.normal(size=(L, C))
    u = rng.normal(size=L)
    lam = 0.4
    target = np.array([1.0, 0.0])
    C2, Q2 = local_wrls_step_numpy(Cmat, Q, u, lam, target)
    # in_features=8 => d_ext=9 >= L=5
    m = VSRPAnyaEFS(8, 1, 2, binary=True, seed=0, proj_dim=L, dtype=torch.float32)
    m._classes = np.array([0, 1])
    m._reset_state()
    m._classes = np.array([0, 1])
    m._bootstrap_first_cloud(np.zeros(8))
    m._P[0] = Cmat.copy()
    with torch.no_grad():
        m.consequent.data = torch.as_tensor(Q[None, ...], dtype=torch.float32)
    m._fwrls_update(np.array([lam]), u[None, :], target)
    assert np.allclose(m._P[0], C2)
    assert np.allclose(m.consequent.detach().cpu().numpy()[0], Q2)


def test_new_cloud_Q_zero_C_p0I():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, p_init=500.0, dtype=torch.float32)
    m.partial_fit(np.array([[0.0, 0.0]]), np.array([0]), classes=[0, 1])
    assert torch.allclose(m.consequent, torch.zeros_like(m.consequent))
    assert np.allclose(m._P[0], 500.0 * np.eye(m.proj_dim))
    assert float(m.radii[0]) == pytest.approx(1.0)


def test_covariance_finite_symmetric():
    m = VSRPAnyaEFS(3, 1, 2, binary=True, seed=0, dtype=torch.float32)
    X, y = _toy_binary(30, 3)
    m.fit(X, y)
    for P in m._P:
        assert np.isfinite(P).all()
        assert np.allclose(P, P.T, atol=1e-8)


# ---------------------------------------------------------------------------
# Fit / inference behaviour
# ---------------------------------------------------------------------------
def test_fit_processes_each_sample_once_with_bootstrap():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
    y = np.array([0, 1, 0])
    with mock.patch.object(m, "_bootstrap_first_cloud", wraps=m._bootstrap_first_cloud) as boot:
        with mock.patch.object(m, "_online_step", wraps=m._online_step) as stepped:
            m.fit(X, y)
            assert boot.call_count == 1
            assert stepped.call_count == 2
    assert m._n_seen == 3
    assert m._train_step_index == 2


def test_no_optimizer_scheduler_backward_reconstruction():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    assert m.uses_custom_fit and not m.supports_backprop_training
    assert not m.uses_generic_optimizer
    assert m.uses_reconstruction is False
    X, y = _toy_binary(15, 2)
    with mock.patch("torch.optim.Adam", side_effect=AssertionError("Adam")):
        with mock.patch("torch.optim.SGD", side_effect=AssertionError("SGD")):
            m.fit(X, y)
    scores, recon = m(torch.as_tensor(X, dtype=torch.float32))
    assert scores.shape == (15, 2)
    assert torch.equal(recon, torch.as_tensor(X, dtype=torch.float32))


def test_inference_mutation_free_and_deterministic():
    m = VSRPAnyaEFS(3, 1, 2, binary=True, seed=7, dtype=torch.float32)
    X, y = _toy_binary(20, 3, seed=7)
    w = SklearnVSRPAnyaEFSWrapper(m)
    w.fit(X, y)
    snap = copy.deepcopy(m.snapshot_state())
    p1 = w.predict(X)
    p2 = w.predict(X)
    assert np.array_equal(p1, p2)
    snap2 = m.snapshot_state()
    assert snap["n_seen"] == snap2["n_seen"]
    assert snap["train_step_index"] == snap2["train_step_index"]
    assert np.allclose(snap["Q"], snap2["Q"])
    for a, b in zip(snap["P"], snap2["P"]):
        assert np.allclose(a, b)


def test_fixed_inference_projection_order_and_batch_invariant():
    X, y = _toy_binary(16, 3, seed=3)
    m = VSRPAnyaEFS(3, 1, 2, binary=True, seed=3, dtype=torch.float32)
    w = SklearnVSRPAnyaEFSWrapper(m, batch_size=4)
    w.fit(X, y)
    assert "deterministic_fixed_inference" in m.inference_projection_adaptation
    scores = w.decision_function(X)
    # Singleton vs batch
    scores_one = np.vstack([w.decision_function(X[i : i + 1]) for i in range(len(X))])
    assert np.allclose(scores, scores_one, atol=1e-5)
    # Reorder
    perm = np.array([3, 0, 5, 1, 2, 7, 6, 4, 8, 9, 10, 11, 12, 13, 14, 15])
    scores_perm = w.decision_function(X[perm])
    assert np.allclose(scores_perm, scores[perm], atol=1e-5)
    # Batch size change
    w.set_params(batch_size=1)
    assert np.allclose(w.decision_function(X), scores, atol=1e-5)
    w.set_params(batch_size=16)
    assert np.allclose(w.decision_function(X), scores, atol=1e-5)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def test_checkpoint_prediction_roundtrip():
    m = VSRPAnyaEFS(3, 1, 2, binary=True, seed=0, dtype=torch.float32)
    X, y = _toy_binary(25, 3)
    w = SklearnVSRPAnyaEFSWrapper(m)
    w.fit(X, y)
    pred = w.predict(X)
    m2 = VSRPAnyaEFS(3, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m2.restore_checkpoint(m.state_dict(), strict=True)
    w2 = SklearnVSRPAnyaEFSWrapper(m2)
    assert np.array_equal(w2.predict(X), pred)


def test_checkpoint_partial_fit_continuation():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    X, y = _toy_binary(30, 2)
    m.partial_fit(X[:10], y[:10], classes=[0, 1])
    m2 = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m2.restore_checkpoint(m.state_dict(), strict=True)
    m2.partial_fit(X[10:20], y[10:20])
    m.partial_fit(X[10:20], y[10:20])
    assert m.rules_count == m2.rules_count
    assert m._n_seen == m2._n_seen
    assert np.allclose(
        m.consequent.detach().cpu().numpy(),
        m2.consequent.detach().cpu().numpy(),
        atol=1e-5,
    )


def _assert_snapshots_equal(a, b):
    assert a["rules_count"] == b["rules_count"]
    assert a["n_seen"] == b["n_seen"]
    assert a["train_step_index"] == b["train_step_index"]
    assert a["projection_mode"] == b["projection_mode"]
    assert a["inference_seed"] == b["inference_seed"]
    assert np.allclose(a["centers"], b["centers"])
    assert np.allclose(a["foci"], b["foci"])
    assert np.allclose(a["radii"], b["radii"])
    assert np.allclose(a["scatters"], b["scatters"])
    assert np.allclose(a["supports"], b["supports"])
    assert np.allclose(a["avg_sq"], b["avg_sq"])
    assert np.allclose(a["Q"], b["Q"])
    assert len(a["P"]) == len(b["P"])
    for pa, pb in zip(a["P"], b["P"]):
        assert np.allclose(pa, pb)
    if a["classes"] is None:
        assert b["classes"] is None
    else:
        assert np.array_equal(np.asarray(a["classes"]), np.asarray(b["classes"]))
    if a["global_mean"] is None:
        assert b["global_mean"] is None
    else:
        assert np.allclose(a["global_mean"], b["global_mean"])
    assert a["global_avg_sq"] == pytest.approx(b["global_avg_sq"])
    if a.get("inference_R") is None:
        assert b.get("inference_R") is None
    else:
        assert np.allclose(a["inference_R"], b["inference_R"])


def test_strict_rejects_missing_extra_state():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    st = dict(m.state_dict())
    del st["_extra_state"]
    dest = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    dest.partial_fit(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0, 1]), classes=[0, 1])
    before = dest.snapshot_state()
    before_fitted = dest.is_fitted_
    with pytest.raises(ValueError, match="_extra_state"):
        dest.load_state_dict(st, strict=True)
    _assert_snapshots_equal(before, dest.snapshot_state())
    assert dest.is_fitted_ == before_fitted


def test_strict_rejects_missing_covariance_matrices():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    st = dict(m.state_dict())
    extra = dict(st["_extra_state"])
    del extra["P"]
    st["_extra_state"] = extra
    dest = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="covariance|extra keys|P"):
        dest.load_state_dict(st, strict=True)


def test_strict_rejects_missing_classes():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    st = dict(m.state_dict())
    extra = dict(st["_extra_state"])
    del extra["classes"]
    st["_extra_state"] = extra
    dest = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="classes|extra keys"):
        dest.load_state_dict(st, strict=True)


def test_strict_rejects_unexpected_keys():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    st = dict(m.state_dict())
    st["totally_unexpected_key"] = torch.zeros(1)
    dest = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="unexpected"):
        dest.load_state_dict(st, strict=True)


def test_strict_rejects_missing_required_tensor_keys():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    st = dict(m.state_dict())
    del st["foci"]
    dest = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    dest.partial_fit(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0, 1]), classes=[0, 1])
    before = dest.snapshot_state()
    with pytest.raises(ValueError, match="missing"):
        dest.load_state_dict(st, strict=True)
    _assert_snapshots_equal(before, dest.snapshot_state())


def test_strict_rejects_incompatible_proj_dim():
    m2 = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, proj_dim=2, dtype=torch.float32)
    m2.fit(*_toy_binary(10, 2))
    m3 = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, proj_dim=3, dtype=torch.float32)
    with pytest.raises(ValueError, match="proj_dim"):
        m3.load_state_dict(m2.state_dict(), strict=True)


def test_strict_rejects_incompatible_out_features():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    m4 = VSRPAnyaEFS(2, 1, 3, binary=False, seed=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="out_features"):
        m4.load_state_dict(m.state_dict(), strict=True)


def test_strict_rejects_invalid_class_count():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    st = dict(m.state_dict())
    extra = dict(st["_extra_state"])
    extra["classes"] = [0, 1, 2]
    st["_extra_state"] = extra
    dest = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="two classes|class count"):
        dest.load_state_dict(st, strict=True)


def test_strict_rejects_nonfinite_state():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    st = dict(m.state_dict())
    st["centers"] = st["centers"].clone()
    st["centers"][0, 0] = float("nan")
    dest = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="finite|non-finite"):
        dest.load_state_dict(st, strict=True)


def test_failed_strict_load_leaves_destination_unchanged():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(12, 2))
    dest = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    dest.partial_fit(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0, 1]), classes=[0, 1])
    before = dest.snapshot_state()
    before_fitted = dest.is_fitted_
    before_p = [p.copy() for p in dest._P]
    before_meta = {
        "p_init": dest.p_init,
        "max_rules": dest.max_rules,
        "compression_ratio": dest.compression_ratio,
        "projection_mode": dest.projection_mode,
        "inference_seed": dest.inference_seed,
        "seed": dest.seed,
        "sigma_eps": dest.sigma_eps,
        "density_epsilon": dest.density_epsilon,
        "rules_hint": dest.rules_hint,
        "compression_source": dest.compression_source,
    }

    bad = dict(m.state_dict())
    del bad["_extra_state"]
    with pytest.raises(ValueError):
        dest.load_state_dict(bad, strict=True)

    _assert_snapshots_equal(before, dest.snapshot_state())
    assert dest.is_fitted_ == before_fitted
    for a, b in zip(before_p, dest._P):
        assert np.allclose(a, b)
    assert dest.p_init == before_meta["p_init"]
    assert dest.max_rules == before_meta["max_rules"]
    assert dest.compression_ratio == before_meta["compression_ratio"]
    assert dest.projection_mode == before_meta["projection_mode"]
    assert dest.inference_seed == before_meta["inference_seed"]
    assert dest.seed == before_meta["seed"]
    assert dest.sigma_eps == pytest.approx(before_meta["sigma_eps"])
    assert dest.density_epsilon == pytest.approx(before_meta["density_epsilon"])
    assert dest.rules_hint == before_meta["rules_hint"]
    assert dest.compression_source == before_meta["compression_source"]


def test_nonstrict_tensor_only_legacy_load_warns_and_unfitted():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    st = dict(m.state_dict())
    del st["_extra_state"]
    dest = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    with pytest.warns(RuntimeWarning, match="tensor-only|no extra state"):
        dest.load_state_dict(st, strict=False)
    assert dest.is_fitted_ is False
    assert dest._P == []
    with pytest.raises(Exception):
        dest(torch.zeros(1, 2))


def test_nonstrict_legacy_missing_foci_warns():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    st = dict(m.state_dict())
    del st["foci"]
    dest = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    with pytest.warns(RuntimeWarning, match="foci"):
        dest.load_state_dict(st, strict=False)
    assert dest.rules_count == m.rules_count


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
def test_wrapper_not_fitted_and_labels_and_proba():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    w = SklearnVSRPAnyaEFSWrapper(m)
    with pytest.raises(NotFittedError):
        w.predict(np.zeros((1, 2)))
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    y = np.array(["neg", "pos", "neg"], dtype=object)
    w.fit(X, y)
    pred = w.predict(X)
    assert set(pred).issubset({"neg", "pos"})
    assert w.predict_proba(X).shape == (3, 2)
    w.set_params(model=VSRPAnyaEFS(2, 1, 2, binary=True, seed=1, dtype=torch.float32))
    assert w.is_fitted_ is False
    with pytest.raises(TypeError, match="unexpected"):
        w.fit(X, y, learning_rate=0.1)


def test_train_only_minmax_scaling_helper_pattern():
    X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]], dtype=np.float64)
    X_te = np.array([[2.5, 15.0]], dtype=np.float64)
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    Xtr = scaler.fit_transform(X)
    Xte = scaler.transform(X_te)
    assert Xtr.min() >= -1.0 - 1e-9 and Xtr.max() <= 1.0 + 1e-9
    assert Xte.shape == (1, 2)


def test_flags_present():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, dtype=torch.float32)
    assert m.uses_custom_fit is True
    assert m.supports_backprop_training is False
    assert m.uses_generic_optimizer is False
    assert m.uses_reconstruction is False
    assert m.uses_reconstruction_loss is False
    assert m.uses_vsrp_anya_protocol is True
    assert m.required_input_scaling == "minmax_m1_1"
    assert m.training_mode == "online_anya_vsrp_local_wrls"
    assert m.projection_mode == PROJECTION_PAPER_DYNAMIC
    assert m.unused_compatibility_parameter == "psi"


def test_fixed_per_cloud_mode_named_adaptation():
    m = VSRPAnyaEFS(
        2,
        1,
        2,
        binary=True,
        seed=0,
        projection_mode=PROJECTION_FIXED_PER_CLOUD,
        dtype=torch.float32,
    )
    m.fit(*_toy_binary(15, 2))
    assert m.R.shape[0] == m.rules_count



def test_exact_partial_fit_continuation_after_strict_load():
    """fit->checkpoint->load->partial_fit == uninterrupted fit->partial_fit."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4)).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)
    X1, y1 = X[:30], y[:30]
    X2, y2 = X[30:], y[30:]

    model_full = VSRPAnyaEFS(
        4, 1, 2, binary=True, seed=11, sigma_eps=1e-6, p_init=500.0, dtype=torch.float32
    )
    model_full.fit(X1, y1)
    model_full.partial_fit(X2, y2)

    model_part = VSRPAnyaEFS(
        4, 1, 2, binary=True, seed=11, sigma_eps=1e-6, p_init=500.0, dtype=torch.float32
    )
    model_part.fit(X1, y1)
    saved = model_part.state_dict()

    model_loaded = VSRPAnyaEFS(
        4,
        1,
        2,
        binary=True,
        seed=999,
        inference_seed=12345,
        sigma_eps=1e-3,
        density_epsilon=1e-3,
        p_init=100.0,
        dtype=torch.float32,
    )
    assert model_loaded.seed == 999
    assert model_loaded.sigma_eps == pytest.approx(1e-3)
    model_loaded.load_state_dict(saved, strict=True)

    assert model_loaded.seed == model_part.seed == 11
    assert model_loaded.sigma_eps == pytest.approx(model_part.sigma_eps)
    assert model_loaded.density_epsilon == pytest.approx(model_part.density_epsilon)
    assert model_loaded.p_init == pytest.approx(500.0)
    assert model_loaded.inference_seed == model_part.inference_seed

    model_loaded.partial_fit(X2, y2)

    assert model_loaded.rules_count == model_full.rules_count
    assert model_loaded._n_seen == model_full._n_seen
    assert model_loaded._train_step_index == model_full._train_step_index
    assert model_loaded._next_rule_index == model_full._next_rule_index
    assert model_loaded.seed == model_full.seed
    assert model_loaded.sigma_eps == pytest.approx(model_full.sigma_eps)
    assert model_loaded.density_epsilon == pytest.approx(model_full.density_epsilon)
    assert model_loaded.p_init == pytest.approx(model_full.p_init)
    assert model_loaded.projection_mode == model_full.projection_mode
    assert list(model_loaded._cloud_updated_flags) == list(model_full._cloud_updated_flags)
    assert np.array_equal(np.asarray(model_loaded._classes), np.asarray(model_full._classes))
    assert np.allclose(model_loaded.centers.detach().cpu().numpy(), model_full.centers.detach().cpu().numpy())
    assert np.allclose(model_loaded.foci.detach().cpu().numpy(), model_full.foci.detach().cpu().numpy())
    assert np.allclose(model_loaded.radii.detach().cpu().numpy(), model_full.radii.detach().cpu().numpy())
    assert np.allclose(model_loaded.scatters.detach().cpu().numpy(), model_full.scatters.detach().cpu().numpy())
    assert np.allclose(model_loaded.supports.detach().cpu().numpy(), model_full.supports.detach().cpu().numpy())
    assert np.allclose(model_loaded.avg_sq.detach().cpu().numpy(), model_full.avg_sq.detach().cpu().numpy())
    assert np.allclose(
        model_loaded.consequent.detach().cpu().numpy(),
        model_full.consequent.detach().cpu().numpy(),
    )
    for a, b in zip(model_loaded._P, model_full._P):
        assert np.allclose(a, b)
    assert np.allclose(model_loaded._global_mean, model_full._global_mean)
    assert model_loaded._global_avg_sq == pytest.approx(model_full._global_avg_sq)

    w_full = SklearnVSRPAnyaEFSWrapper(model_full)
    w_loaded = SklearnVSRPAnyaEFSWrapper(model_loaded)
    assert np.array_equal(w_full.predict(X), w_loaded.predict(X))
    assert np.allclose(w_full.decision_function(X), w_loaded.decision_function(X), atol=1e-5)


def test_failed_strict_load_restores_seed_and_epsilons():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, sigma_eps=1e-6, dtype=torch.float32)
    m.fit(*_toy_binary(12, 2))
    dest = VSRPAnyaEFS(
        2, 1, 2, binary=True, seed=7, sigma_eps=2e-6, density_epsilon=2e-6, dtype=torch.float32
    )
    dest.partial_fit(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0, 1]), classes=[0, 1])
    before = {
        "seed": dest.seed,
        "sigma_eps": dest.sigma_eps,
        "density_epsilon": dest.density_epsilon,
        "rules_hint": dest.rules_hint,
        "p_init": dest.p_init,
        "max_rules": dest.max_rules,
        "compression_ratio": dest.compression_ratio,
        "compression_source": dest.compression_source,
        "snap": dest.snapshot_state(),
        "fitted": dest.is_fitted_,
    }
    bad = dict(m.state_dict())
    extra = dict(bad["_extra_state"])
    extra["density_epsilon"] = -1.0
    bad["_extra_state"] = extra
    with pytest.raises(ValueError, match="density_epsilon"):
        dest.load_state_dict(bad, strict=True)
    assert dest.seed == before["seed"]
    assert dest.sigma_eps == pytest.approx(before["sigma_eps"])
    assert dest.density_epsilon == pytest.approx(before["density_epsilon"])
    assert dest.rules_hint == before["rules_hint"]
    assert dest.p_init == pytest.approx(before["p_init"])
    assert dest.max_rules == before["max_rules"]
    assert dest.compression_ratio == before["compression_ratio"]
    assert dest.compression_source == before["compression_source"]
    assert dest.is_fitted_ == before["fitted"]
    _assert_snapshots_equal(before["snap"], dest.snapshot_state())


def test_nonstrict_missing_seed_epsilon_warns_keeps_constructor_values():
    m = VSRPAnyaEFS(2, 1, 2, binary=True, seed=0, sigma_eps=1e-6, dtype=torch.float32)
    m.fit(*_toy_binary(10, 2))
    st = dict(m.state_dict())
    extra = dict(st["_extra_state"])
    del extra["seed"]
    del extra["sigma_eps"]
    del extra["density_epsilon"]
    st["_extra_state"] = extra
    dest = VSRPAnyaEFS(
        2, 1, 2, binary=True, seed=77, sigma_eps=5e-4, density_epsilon=5e-4, dtype=torch.float32
    )
    with pytest.warns(RuntimeWarning, match="Exact online continuation is not guaranteed"):
        dest.load_state_dict(st, strict=False)
    assert dest.seed == 77
    assert dest.sigma_eps == pytest.approx(5e-4)
    assert dest.density_epsilon == pytest.approx(5e-4)


def test_e2e_binary_multiclass_checkpoint_partial():
    Xb, yb = _toy_binary(40, 4)
    mb = VSRPAnyaEFS(4, 1, 2, binary=True, seed=0, dtype=torch.float32)
    wb = SklearnVSRPAnyaEFSWrapper(mb)
    wb.fit(Xb, yb)
    assert wb.predict(Xb).shape == (40,)

    Xm, ym = _toy_multi(50, 4)
    mm = VSRPAnyaEFS(4, 1, 3, binary=False, seed=0, dtype=torch.float32)
    wm = SklearnVSRPAnyaEFSWrapper(mm)
    wm.fit(Xm, ym)
    assert wm.predict_proba(Xm).shape == (50, 3)

    mm2 = VSRPAnyaEFS(4, 1, 3, binary=False, seed=0, dtype=torch.float32)
    mm2.restore_checkpoint(mm.state_dict(), strict=True)
    assert np.array_equal(
        SklearnVSRPAnyaEFSWrapper(mm2).predict(Xm), wm.predict(Xm)
    )
    mm.partial_fit(Xm[:5], ym[:5])
