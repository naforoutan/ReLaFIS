"""VSRP-AnYa-EFS paper protocol helpers (alias module).

Modes
-----
- ``paper_table_mode``: evaluate each compression ratio independently
  (matches paper tables; no test-fold selection of k).
- ``leakage_safe_selection_mode``: select compression ratio by inner CV on the
  outer training fold only, then evaluate once on the outer test fold.

Preprocessing: train-fold MinMaxScaler feature_range=(-1, 1).
"""

from __future__ import annotations

# Relative import avoids loading experiments.__init__ side effects when this
# file is loaded via importlib in tests.
from experiments.protocols.vsrp_anya import (  # noqa: F401
    ADVERTISEMENT_COMPRESSION_FACTORS,
    DATASET_PROFILES,
    GISETTE_COMPRESSION_FACTORS,
    STANDARD_COMPRESSION_FACTORS,
    run_vsrp_anya_protocol,
    run_vsrp_single_split,
    run_vsrp_stratified_cv,
    select_compression_ratio,
)

__all__ = [
    "STANDARD_COMPRESSION_FACTORS",
    "ADVERTISEMENT_COMPRESSION_FACTORS",
    "GISETTE_COMPRESSION_FACTORS",
    "DATASET_PROFILES",
    "select_compression_ratio",
    "run_vsrp_single_split",
    "run_vsrp_stratified_cv",
    "run_vsrp_anya_protocol",
]
