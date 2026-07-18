"""Protocol tests for VSRP-AnYa-EFS paper CV / compression grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.protocols.vsrp_anya import (
    ADVERTISEMENT_COMPRESSION_FACTORS,
    DATASET_PROFILES,
    GISETTE_COMPRESSION_FACTORS,
    STANDARD_COMPRESSION_FACTORS,
    run_vsrp_single_split,
    run_vsrp_stratified_cv,
)
from experiments.protocols import vsrp_anya_efs
from model.vsrp_anya import SklearnVSRPAnyaEFSWrapper, VSRPAnyaEFS

ROOT = Path(__file__).resolve().parents[1]


def test_compression_grids():
    assert STANDARD_COMPRESSION_FACTORS == (2, 3, 4)
    assert ADVERTISEMENT_COMPRESSION_FACTORS == (5, 10, 15)
    assert GISETTE_COMPRESSION_FACTORS == (10, 20, 30, 40, 50)
    assert DATASET_PROFILES["standard"]["n_splits"] == 10
    assert DATASET_PROFILES["gisette"]["n_splits"] == 5
    assert DATASET_PROFILES["advertisement"]["compression_factors"] == (5, 10, 15)


def test_alias_module_exports():
    assert vsrp_anya_efs.STANDARD_COMPRESSION_FACTORS == (2, 3, 4)
    assert callable(vsrp_anya_efs.run_vsrp_anya_protocol)
    text = (ROOT / "experiments/protocols/vsrp_anya_efs.py").read_text()
    assert "paper_table_mode" in text
    assert "leakage_safe_selection_mode" in text


def test_10fold_and_gisette_5fold_protocol_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4)).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)

    result = run_vsrp_stratified_cv(
        VSRPAnyaEFS,
        SklearnVSRPAnyaEFSWrapper,
        X,
        y,
        binary=True,
        seed=0,
        dataset_profile="standard",
        mode="paper_table_mode",
        config={"n_splits": 2, "compression_factors": (2, 3)},
    )
    assert "per_compression_results" in result.extras
    assert 2 in result.extras["per_compression_results"]
    assert 3 in result.extras["per_compression_results"]

    result5 = run_vsrp_stratified_cv(
        VSRPAnyaEFS,
        SklearnVSRPAnyaEFSWrapper,
        X,
        y,
        binary=True,
        seed=0,
        dataset_profile="gisette",
        mode="paper_table_mode",
        config={"n_splits": 2, "compression_factors": (10,)},
    )
    assert result5.extras["dataset_profile"] == "gisette"


def test_leakage_safe_selection_mode():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3)).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)
    result = run_vsrp_stratified_cv(
        VSRPAnyaEFS,
        SklearnVSRPAnyaEFSWrapper,
        X,
        y,
        binary=True,
        seed=0,
        dataset_profile="standard",
        mode="leakage_safe_selection_mode",
        config={"n_splits": 2, "compression_factors": (2, 3), "inner_splits": 2},
    )
    assert result.extras["mode"] == "selected"
    assert "fold_metadata" in result.extras
    for fm in result.extras["fold_metadata"]:
        assert "selected_compression_factor" in fm


def test_single_split_binary_two_outputs_and_minmax():
    rng = np.random.default_rng(2)
    Xtr = rng.normal(size=(30, 3))
    ytr = (Xtr[:, 0] > 0).astype(int)
    Xte = rng.normal(size=(10, 3))
    yte = (Xte[:, 0] > 0).astype(int)
    result = run_vsrp_single_split(
        VSRPAnyaEFS,
        SklearnVSRPAnyaEFSWrapper,
        Xtr,
        ytr,
        Xte,
        yte,
        binary=True,
        seed=0,
        compression_ratio=2,
    )
    assert result.wrapper.model.out_features == 2
    assert result.extras["preprocessing"] == "raw_train_fitted_minmax_m1_1"
    assert result.extras["projection_mode"] == "paper_dynamic"
    assert result.extras["feature_range"] == (-1.0, 1.0)
