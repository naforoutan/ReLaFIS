"""Focused unit tests for the ADMTSK PyTorch reproduction."""

from __future__ import annotations

import math
import warnings
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.admtsk import (
    ADMTSK,
    SklearnADMTSKWrapper,
    adaptive_lambda,
    paper_mse_loss,
    train_admtsk_model,
)
from model.anfis import ANFIS, SklearnAnfisWrapper
from tests.evaluate import Evaluator


def _numpy_dombi(mu: np.ndarray, lam: float) -> np.ndarray:
    """Direct NumPy Dombi (eq. 21). mu: [B, R, D] -> phi: [B, R]."""
    base = np.maximum(1.0 / mu - 1.0, 0.0)
    powered = np.sum(np.power(base, lam), axis=2)
    root = np.power(powered, 1.0 / lam)
    return 1.0 / (1.0 + root)


class _TinyExperiment:
    """Minimal experiment stub for Evaluator smoke tests."""

    def __init__(self, n_samples=40, n_features=4, seed=0):
        rng = np.random.RandomState(seed)
        self._X = rng.rand(n_samples, n_features).astype(np.float32)
        self._y = (self._X[:, 0] + self._X[:, 1] > 1.0).astype(np.int64)
        self.n_folds = 2
        self._fold = 0

    def is_kfold(self):
        return False

    def has_validation_split(self):
        return False

    def resplit(self, session_id=None):
        return None

    def set_fold(self, fold_index):
        self._fold = int(fold_index)

    def train_numpy(self):
        n = int(0.7 * len(self._y))
        return self._X[:n], self._y[:n]

    def test_numpy(self):
        n = int(0.7 * len(self._y))
        return self._X[n:], self._y[n:]

    def get_data(self):
        import pandas as pd

        df = pd.DataFrame(self._X, columns=[f"f{i}" for i in range(self._X.shape[1])])
        return df, pd.Series(self._y, name="target")

    def split_description(self):
        return "tiny synthetic holdout"


# ---------------------------------------------------------------------------
# 1. Adaptive lambda
# ---------------------------------------------------------------------------

def test_adaptive_lambda_reference_values():
    expected = {
        4: 0.5089600,
        1000: 2.5360928,
        10000: 3.3814571,
        100000: 4.2268214,
    }
    for D, ref in expected.items():
        got = adaptive_lambda(D, K=10.0, membership_lower_bound=1.0 / math.e)
        assert got == pytest.approx(ref, rel=1e-6, abs=1e-7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model = ADMTSK(D, rules=3, out_features=2, binary=True)
        assert float(model.lambda_value) == pytest.approx(ref, rel=1e-6, abs=1e-7)


def test_adaptive_lambda_warns_when_below_one():
    with pytest.warns(RuntimeWarning, match="lambda < 1"):
        model = ADMTSK(4, rules=3, out_features=2, binary=True)
    assert float(model.lambda_value) < 1.0


# ---------------------------------------------------------------------------
# 2–3. CGMF
# ---------------------------------------------------------------------------

def test_cgmf_at_center_is_one():
    model = ADMTSK(5, rules=3, out_features=2, binary=True)
    # Use center of rule 1 (= 0.5) on every feature
    x = torch.full((4, 5), 0.5)
    mu = model.membership(x)
    # Rule index 1 has center 0.5
    assert torch.allclose(mu[:, 1, :], torch.ones_like(mu[:, 1, :]), atol=1e-5)


def test_cgmf_lower_bound():
    model = ADMTSK(3, rules=3, out_features=1, binary=True)
    x = torch.full((2, 3), 1e6)  # far from all centers in [0, 1]
    mu = model.membership(x)
    lo = 1.0 / math.e
    assert torch.all(mu >= lo - 1e-5)
    assert torch.all(mu <= 1.0 + 1e-5)
    assert torch.allclose(mu, torch.full_like(mu, lo), atol=1e-4)


# ---------------------------------------------------------------------------
# 4–6. Dombi firing
# ---------------------------------------------------------------------------

def test_dombi_firing_matches_numpy_reference():
    B, R, D = 3, 2, 4
    rng = np.random.RandomState(0)
    # Memberships in (1/e, 1]
    mu_np = lo = 1.0 / math.e
    mu_np = lo + (1.0 - lo) * rng.rand(B, R, D)
    mu_np = np.clip(mu_np, lo, 1.0)
    lam = adaptive_lambda(D)
    ref = _numpy_dombi(mu_np, lam)

    model = ADMTSK(D, rules=R, out_features=2, binary=True)
    mu_t = torch.tensor(mu_np, dtype=torch.float32)
    phi = model.dombi_firing(mu_t).detach().numpy()
    assert np.allclose(phi, ref, rtol=1e-5, atol=1e-6)


def test_firing_strength_is_finite_and_positive():
    model = ADMTSK(8, rules=3, out_features=2, binary=True)
    x = torch.rand(16, 8)
    phi = model.firing_strengths(x)
    assert torch.isfinite(phi).all()
    assert (phi > 0).all()
    assert (phi <= 1.0 + 1e-5).all()


def test_normalized_firing_sums_to_one():
    model = ADMTSK(6, rules=3, out_features=2, binary=True)
    x = torch.rand(10, 6)
    norm = model.normalized_firing_strengths(x)
    assert torch.allclose(norm.sum(dim=1), torch.ones(10), atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Initialization
# ---------------------------------------------------------------------------

def test_parameter_initialization():
    model = ADMTSK(7, rules=3, out_features=2, binary=True)
    expected = torch.tensor([0.0, 0.5, 1.0])
    for d in range(7):
        assert torch.allclose(model.centers[:, d], expected, atol=1e-6)
    assert torch.allclose(model.spreads, torch.ones_like(model.spreads), atol=1e-4)
    assert torch.count_nonzero(model.consequent_weights) == 0
    assert torch.count_nonzero(model.consequent_bias) == 0


# ---------------------------------------------------------------------------
# 8. Forward shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "out_features,binary",
    [(1, True), (2, True), (4, False)],
)
def test_forward_shapes(out_features, binary):
    D, B = 5, 9
    model = ADMTSK(D, rules=3, out_features=out_features, binary=binary)
    x = torch.rand(B, D)
    logits, xd = model(x)
    assert logits.shape == (B, out_features)
    assert xd.shape == x.shape
    assert not model.uses_reconstruction


# ---------------------------------------------------------------------------
# 9. Gradients
# ---------------------------------------------------------------------------

def test_gradients_are_finite():
    model = ADMTSK(4, rules=3, out_features=2, binary=True)
    x = torch.rand(8, 4)
    y = torch.randint(0, 2, (8,))
    y_oh = F.one_hot(y, num_classes=2).float()

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    # Warm up consequents so antecedent grads can become nonzero
    for _ in range(3):
        opt.zero_grad()
        logits, _ = model(x)
        loss = paper_mse_loss(logits, y_oh)
        loss.backward()
        opt.step()

    opt.zero_grad()
    logits, _ = model(x)
    loss = paper_mse_loss(logits, y_oh)
    loss.backward()

    for name, p in [
        ("centers", model.centers),
        ("raw_spreads", model.raw_spreads),
        ("consequent_weights", model.consequent_weights),
        ("consequent_bias", model.consequent_bias),
    ]:
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


# ---------------------------------------------------------------------------
# 10. Chunking
# ---------------------------------------------------------------------------

def test_chunked_and_nonchunked_outputs_match():
    D = 64
    base = ADMTSK(D, rules=3, out_features=2, binary=True, feature_chunk_size=None)
    chunked = ADMTSK(D, rules=3, out_features=2, binary=True, feature_chunk_size=8)
    chunked.load_state_dict(deepcopy(base.state_dict()))
    x = torch.rand(5, D)
    y0, _ = base(x)
    y1, _ = chunked(x)
    assert torch.allclose(y0, y1, rtol=1e-4, atol=1e-5)
    phi0 = base.firing_strengths(x)
    phi1 = chunked.firing_strengths(x)
    assert torch.allclose(phi0, phi1, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 11. Wrapper
# ---------------------------------------------------------------------------

def test_wrapper_predict_proba():
    model = ADMTSK(5, rules=3, out_features=2, binary=True)
    wrap = SklearnADMTSKWrapper(model, batch_size=3)
    X = np.random.RandomState(0).rand(11, 5).astype(np.float32)
    proba = wrap.predict_proba(X)
    assert proba.shape == (11, 2)
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = wrap.predict(X)
    assert pred.shape == (11,)

    model1 = ADMTSK(5, rules=3, out_features=1, binary=True)
    wrap1 = SklearnADMTSKWrapper(model1)
    proba1 = wrap1.predict_proba(X)
    assert proba1.shape == (11, 2)
    assert np.allclose(proba1.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 12. No reconstruction for ADMTSK
# ---------------------------------------------------------------------------

def test_no_reconstruction_loss_for_admtsk():
    experiment = _TinyExperiment()
    device = torch.device("cpu")
    configs = {
        "ADMTSK": {
            "model_class": ADMTSK,
            "wrapper_class": SklearnADMTSKWrapper,
            "params": {
                "rules": 3,
                "out_features": 2,
                "binary": True,
                "validate_input_range": False,
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
        "alpha": 1.0,
        "min_alpha": 0.0,
    }

    ev1 = Evaluator(
        experiment=experiment,
        model_configs=configs,
        learning_params=learning_params,
        device=device,
        binary=True,
        noise_levels=[0.0],
        n_runs=1,
        random_state=0,
        use_noise=False,
        use_early_stopping=False,
    )
    # Patch learning alpha mid-flight via two evaluators with different alpha
    learning_hi = dict(learning_params)
    learning_hi["alpha"] = 100.0
    ev2 = Evaluator(
        experiment=experiment,
        model_configs=configs,
        learning_params=learning_hi,
        device=device,
        binary=True,
        noise_levels=[0.0],
        n_runs=1,
        random_state=0,
        use_noise=False,
        use_early_stopping=False,
    )
    r1 = ev1.evaluate(verbose=False)
    r2 = ev2.evaluate(verbose=False)
    # Same seed + no recon term ⇒ identical test accuracy
    assert r1["test_acc"].iloc[0] == pytest.approx(r2["test_acc"].iloc[0], abs=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = ADMTSK(4, rules=3, out_features=2)
    assert not hasattr(m, "relaxation_rate")
    assert m.linguistic_richness() == 0.0


# ---------------------------------------------------------------------------
# 13. Existing model evaluator defaults unchanged
# ---------------------------------------------------------------------------

def test_existing_model_evaluator_behavior_is_unchanged():
    """Lightweight check that default loss/recon/OneCycle path still runs."""
    experiment = _TinyExperiment(n_features=4)
    device = torch.device("cpu")
    configs = {
        "Anfis": {
            "model_class": ANFIS,
            "wrapper_class": SklearnAnfisWrapper,
            "params": {"rules": 2, "drop_out_p": 0.0},
            # no training key ⇒ defaults: loss_mode=default, recon=True, onecycle
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
    assert len(results) == 1
    assert np.isfinite(results["test_acc"].iloc[0])


# ---------------------------------------------------------------------------
# Checkpoint / state_dict engineering
# ---------------------------------------------------------------------------

def _admtsk(D=8, rules=3, out_features=2, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ADMTSK(D, rules=rules, out_features=out_features, binary=True, **kwargs)


def test_state_dict_has_no_duplicate_registered_keys():
    model = _admtsk()
    keys = set(model.state_dict().keys())
    for required in (
        "centers",
        "raw_spreads",
        "consequent_weights",
        "consequent_bias",
        "lambda_value",
    ):
        assert required in keys
    for forbidden in ("mean", "raw_std", "consequent_weight", "_lambda_adaptive"):
        assert forbidden not in keys
    assert model.mean is model.centers
    assert model.raw_std is model.raw_spreads
    assert model.consequent_weight is model.consequent_weights
    assert model._lambda_adaptive is model.lambda_value


def test_state_dict_round_trip():
    a = _admtsk()
    with torch.no_grad():
        a.centers.add_(0.01)
        a.raw_spreads.add_(0.05)
        a.consequent_weights.add_(0.1)
        a.consequent_bias.add_(0.2)
    b = _admtsk()
    incompatible = b.load_state_dict(a.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    x = torch.rand(5, a.in_features)
    ya, _ = a(x)
    yb, _ = b(x)
    assert torch.allclose(ya, yb, atol=1e-6)


def test_legacy_alias_checkpoint_loads_strictly():
    model = _admtsk()
    sd = model.state_dict()
    legacy = {
        "mean": sd["centers"].clone(),
        "raw_std": sd["raw_spreads"].clone(),
        "consequent_weight": sd["consequent_weights"].clone(),
        "consequent_bias": sd["consequent_bias"].clone(),
        "_lambda_adaptive": sd["lambda_value"].clone(),
        "_extra_state": sd["_extra_state"],
    }
    with torch.no_grad():
        legacy["mean"].fill_(0.25)
        legacy["raw_std"].fill_(0.5)
        legacy["consequent_weight"].fill_(0.1)
        legacy["consequent_bias"].fill_(-0.2)

    loaded = _admtsk()
    incompatible = loaded.load_state_dict(legacy, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.allclose(loaded.centers, legacy["mean"])
    assert torch.allclose(loaded.raw_spreads, legacy["raw_std"])
    assert torch.allclose(loaded.consequent_weights, legacy["consequent_weight"])
    assert torch.allclose(loaded.lambda_value, legacy["_lambda_adaptive"])
    out, _ = loaded(torch.rand(4, loaded.in_features))
    assert torch.isfinite(out).all()


def test_legacy_std_conversion():
    model = _admtsk(min_spread=1e-6)
    sd = model.state_dict()
    legacy_std = torch.full_like(sd["raw_spreads"], 1.5)
    legacy = {
        "centers": sd["centers"].clone(),
        "std": legacy_std.clone(),
        "consequent_weights": sd["consequent_weights"].clone(),
        "consequent_bias": sd["consequent_bias"].clone(),
        "lambda_value": sd["lambda_value"].clone(),
        "_extra_state": sd["_extra_state"],
    }
    loaded = _admtsk(min_spread=1e-6)
    with pytest.warns(RuntimeWarning, match="legacy ADMTSK 'std'"):
        incompatible = loaded.load_state_dict(legacy, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.allclose(loaded.spreads, legacy_std, atol=1e-4)


def test_wrapper_inference_is_genuinely_batched():
    model = _admtsk(D=5, out_features=2)
    wrap = SklearnADMTSKWrapper(model, batch_size=3)
    X = np.random.RandomState(0).rand(11, 5).astype(np.float32)
    seen = []

    def _hook(_module, inputs, _output):
        seen.append(int(inputs[0].shape[0]))

    handle = model.register_forward_hook(_hook)
    try:
        proba = wrap.predict_proba(X)
    finally:
        handle.remove()

    assert seen
    assert all(s <= 3 for s in seen)
    assert sum(seen) == 11
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (11, 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    pred = wrap.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (11,)

    with torch.no_grad():
        full = model(torch.as_tensor(X, dtype=torch.float32))[0]
        ref = torch.softmax(full, dim=1).numpy()
    assert np.allclose(proba, ref, atol=1e-5)


def test_wrapper_empty_input():
    model = _admtsk(D=5, out_features=2)
    wrap = SklearnADMTSKWrapper(model, batch_size=3)
    X_empty = np.empty((0, 5), dtype=np.float32)
    proba = wrap.predict_proba(X_empty)
    pred = wrap.predict(X_empty)
    assert proba.shape == (0, 2)
    assert pred.shape == (0,)


def test_wrapper_batch_size_validation():
    model = _admtsk(D=4)
    with pytest.raises(ValueError, match="batch_size"):
        SklearnADMTSKWrapper(model, batch_size=0)
    wrap = SklearnADMTSKWrapper(model, batch_size=2)
    with pytest.raises(ValueError, match="batch_size"):
        wrap.set_params(batch_size=-1)


# ---------------------------------------------------------------------------
# CPU DataLoader training
# ---------------------------------------------------------------------------

def test_train_admtsk_model_cpu_dataloader():
    model = _admtsk(D=6, out_features=2, paper_mode=True)
    rng = np.random.RandomState(1)
    X = rng.rand(24, 6).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int64)
    train_admtsk_model(
        model,
        X,
        y,
        learning_rate=0.01,
        batch_fraction=0.25,
        epochs=3,
        random_state=0,
        device="cpu",
    )
    assert model.is_fitted_
    assert model.training_metadata["scheduler"] is None
    assert model.training_metadata["loss"] == "paper_one_hot_mse"
    assert model.training_metadata["batch_size"] == max(1, round(0.25 * 24))
    for p in model.parameters():
        assert torch.isfinite(p).all()
    wrap = SklearnADMTSKWrapper(model)
    pred = wrap.predict(X)
    assert pred.shape == (24,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_admtsk_dataset_stays_on_cpu_when_model_on_cuda(monkeypatch):
    """Full training tensors must not be created on CUDA."""
    model = _admtsk(D=6, out_features=2).cuda()
    created_devices = []
    real_as_tensor = torch.as_tensor

    def tracking_as_tensor(*args, **kwargs):
        t = real_as_tensor(*args, **kwargs)
        created_devices.append(str(t.device))
        return t

    monkeypatch.setattr(torch, "as_tensor", tracking_as_tensor)
    rng = np.random.RandomState(2)
    X = rng.rand(16, 6).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int64)
    train_admtsk_model(
        model, X, y, epochs=1, batch_fraction=0.5, learning_rate=0.01, device="cuda"
    )
    assert created_devices, "expected torch.as_tensor to build the CPU feature matrix"
    assert all(d.startswith("cpu") for d in created_devices)


def test_paper_mode_requires_cgmf_epsilon():
    with pytest.raises(ValueError, match="membership_lower_bound"):
        ADMTSK(
            8,
            rules=3,
            out_features=2,
            binary=True,
            paper_mode=True,
            membership_lower_bound=0.1,
        )
