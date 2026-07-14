#!/usr/bin/env python3
"""Optional checkpoint-based recomputation of HabermanBoundary grids.

Loads the saved checkpoint + scaler + local Haberman CSV, runs inference only
(no training), and regenerates the figure through ``plot_interpretability_2d``.

Primary reproduction remains the model-free script
``scripts/reproduce_haberman_boundary.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.giftshifter import GIFTSHIFTER
from utils.decision_boundary import plot_interpretability_2d


def _load_local_haberman(seed: int):
    df = pd.read_csv(ROOT / "data/haberman.data", header=None)
    y = df.iloc[:, 3].to_numpy()
    if set(np.unique(y).tolist()) == {1, 2}:
        y = (y == 2).astype(np.int64)
    else:
        y = y.astype(np.int64)
    X = df.iloc[:, :3].to_numpy(dtype=np.float64)
    feature_names = ["Age", "Year", "Positive Axillary Nodes"]
    idx = np.arange(len(X))
    X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(
        X, y, idx, test_size=0.3, random_state=seed, stratify=y,
    )
    return X_train, y_train, X_test, y_test, feature_names, i_train, i_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--stem", default="HabermanBoundary")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write a new HabermanBoundary run (default: sibling _recomputed)",
    )
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    stem = args.stem

    meta_path = artifact_dir / f"{stem}_metadata.json"
    ckpt_path = artifact_dir / f"{stem}_checkpoint.pt"
    scaler_path = artifact_dir / f"{stem}_scaler.joblib"
    cfg_path = artifact_dir / f"{stem}_plot_config.json"
    for p in (meta_path, ckpt_path, scaler_path, cfg_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    with meta_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    scaler = joblib.load(scaler_path)

    seed = int(meta.get("seed", 42))
    X_train_raw, y_train, _, _, feature_names, i_train, i_test = _load_local_haberman(seed)
    X_train = scaler.transform(X_train_raw)

    mcfg = ckpt.get("model_config") or {}
    model = GIFTSHIFTER(
        in_features=int(mcfg["in_features"]),
        rules=int(mcfg["rules"]),
        out_features=int(mcfg["out_features"]),
        binary=bool(mcfg["binary"]),
        drop_out_p=float(mcfg.get("drop_out_p", 0.3)),
        dtype=torch.float32,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out = args.out_dir or (artifact_dir.parent / f"{artifact_dir.name}_recomputed")
    out.mkdir(parents=True, exist_ok=True)
    feature_x = int(cfg.get("feature_x", 2))
    feature_y = int(cfg.get("feature_y", 0))

    plot_interpretability_2d(
        model, X_train, y_train,
        feature_x=feature_x, feature_y=feature_y,
        feature_names=feature_names,
        class_names=cfg.get("legend_labels", {}).get("classes"),
        axis_scale=cfg.get("axis_scale", "standardized"),
        save_path=str(out / stem),
        scaler=scaler,
        seed=seed,
        run_index=meta.get("run_index", 0),
        train_indices=i_train,
        test_indices=i_test,
        source_split="train",
        overwrite_artifacts=True,
        metadata={
            "dataset_name": "Haberman",
            "model_name": "GIFTSHIFTER",
            "notes": "Recomputed from checkpoint (no training)",
            "source_artifact_dir": str(artifact_dir),
        },
    )
    print(f"Recomputed figure written under {out}")


if __name__ == "__main__":
    main()
