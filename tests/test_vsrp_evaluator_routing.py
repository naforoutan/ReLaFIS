"""Evaluator custom-fit routing tests for VSRP-AnYa-EFS."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch

from model.vsrp_anya import SklearnVSRPAnyaEFSWrapper, VSRPAnyaEFS
from tests.evaluate import Evaluator


def test_evaluator_custom_fit_routing_no_optimizer_backward():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    # Minimal fake session: bypass pycaret by constructing Evaluator carefully.
    # Use a lightweight stub of Evaluator internals via direct _train_and_evaluate.
    ev = object.__new__(Evaluator)
    ev.device = torch.device("cpu")
    ev.binary = True
    ev.n_classes = 2
    ev.n_features = 4
    ev.noise_type = "gaussian"
    ev.X_train = X[:30]
    ev.X_test = X[30:]
    ev.y_train = y[:30]
    ev.y_test = y[30:]
    ev.learning_params = {"batch_size": 16, "lr": 0.01, "max_lr": 0.01, "epochs": 2}

    def _refresh(*args, **kwargs):
        return None

    ev._refresh_split = _refresh
    ev._apply_noise = lambda data, std, ntype: data

    fit_calls = {"n": 0}
    real_fit = SklearnVSRPAnyaEFSWrapper.fit

    def spy_fit(self, X, y, **kwargs):
        fit_calls["n"] += 1
        return real_fit(self, X, y, **kwargs)

    training = {
        "trainer": "custom_fit",
        "fit_method": "fit",
        "uses_reconstruction": False,
        "scheduler": "none",
    }
    params = {
        "in_features": 4,
        "rules": 1,
        "out_features": 2,
        "binary": True,
        "compression_ratio": 2,
        "proj_dim": None,
        "p_init": 500.0,
        "max_rules": None,
        "projection_mode": "paper_dynamic",
        "seed": 0,
        "drop_out_p": 0.0,
    }

    with mock.patch.object(SklearnVSRPAnyaEFSWrapper, "fit", spy_fit):
        with mock.patch("torch.optim.Adam") as adam:
            with mock.patch("torch.optim.lr_scheduler.OneCycleLR") as sch:
                with mock.patch.object(torch.Tensor, "backward") as backward:
                    out = Evaluator._train_and_evaluate(
                        ev,
                        VSRPAnyaEFS,
                        params,
                        SklearnVSRPAnyaEFSWrapper,
                        noise_std=0.0,
                        run_id=0,
                        run_seed=0,
                        fold_index=0,
                        training_config=training,
                    )
                    adam.assert_not_called()
                    sch.assert_not_called()
                    backward.assert_not_called()

    assert fit_calls["n"] == 1
    train_acc, test_acc, *_rest, wrapper = out
    assert wrapper is not None
    assert hasattr(wrapper, "vsrp_protocol_metadata_")
    meta = wrapper.vsrp_protocol_metadata_
    assert meta["fold_index"] == 0
    assert meta["projection_mode"] == "paper_dynamic"
    assert meta["compression_ratio"] == 2
    assert np.isnan(meta["linguistic_richness"])
    # Prediction must not call partial_fit
    with mock.patch.object(wrapper, "partial_fit") as pf:
        _ = wrapper.predict(ev.X_test)
        pf.assert_not_called()
    assert 0.0 <= float(test_acc) <= 1.0


def test_notebook_config_imports_and_constructs():
    import json
    from pathlib import Path

    nb = json.loads(
        Path("notebooks/report.ipynb").read_text()
    )
    src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "VSRPAnyaEFS" in src
    assert "SklearnVSRPAnyaEFSWrapper" in src
    assert "paper_dynamic" in src
    assert "minmax_m1_1" in src or "MinMax" in src or "[-1, 1]" in src

    # Construct like the notebook config
    binary = True
    n_classes = 2
    run_seed = 0
    model = VSRPAnyaEFS(
        in_features=5,
        rules=1,
        out_features=n_classes if not binary else 2,
        binary=binary,
        drop_out_p=0.0,
        compression_ratio=3,
        proj_dim=None,
        p_init=500.0,
        max_rules=None,
        projection_mode="paper_dynamic",
        seed=run_seed,
        dtype=torch.float32,
    )
    wrap = SklearnVSRPAnyaEFSWrapper(model)
    assert model.uses_custom_fit is True
    assert model.max_rules is None
