"""Focused helpers for Haberman paper-figure reproducibility artifacts."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch

STEM = "HabermanBoundary"
REQUIRED_GRID_KEYS = (
    "xx",
    "yy",
    "dominant_rule_grid_original",
    "dominant_rule_grid_display",
    "predicted_class_grid",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def package_versions() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
    }
    try:
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
    except Exception:
        versions["scikit-learn"] = None
    try:
        import pandas as _pd

        versions["pandas"] = _pd.__version__
    except Exception:
        versions["pandas"] = None
    return versions


def resolve_haberman_run_dir(
    parent: Union[str, Path],
    *,
    seed: Optional[int] = None,
    run_index: Optional[int] = None,
    fold_index: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """Return ``parent/<run_id>/``, appending a timestamp if conflicted."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)

    if seed is not None and run_index is not None:
        run_id = f"seed_{int(seed)}_run_{int(run_index):02d}"
    elif seed is not None and fold_index is not None:
        run_id = f"seed_{int(seed)}_fold_{int(fold_index):02d}"
    elif seed is not None:
        run_id = f"seed_{int(seed)}"
    else:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    out = parent / run_id
    if out.exists() and any(out.iterdir()) and not overwrite:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = parent / f"{run_id}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def extract_model_config(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract real constructor kwargs available on the module."""
    model = model.model if hasattr(model, "model") and hasattr(model.model, "state_dict") else model
    cfg: Dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__,
    }
    for key in (
        "in_features",
        "rules_count",
        "out_features",
        "binary",
        "drop_out_p",
        "rank",
    ):
        if hasattr(model, key):
            val = getattr(model, key)
            if key == "rules_count":
                cfg["rules"] = int(val)
            else:
                cfg[key] = val.item() if hasattr(val, "item") else val
    if "rules" not in cfg and hasattr(model, "rules"):
        cfg["rules"] = int(model.rules)
    return cfg


def cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    model = model.model if hasattr(model, "model") and hasattr(model.model, "state_dict") else model
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def save_haberman_figure_artifacts(
    *,
    output_dir: Union[str, Path],
    fig,
    points_df: pd.DataFrame,
    grids: Mapping[str, np.ndarray],
    plot_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    model: Optional[torch.nn.Module] = None,
    model_config: Optional[Mapping[str, Any]] = None,
    scaler: Any = None,
    optimizer: Any = None,
    preprocessing_bundle: Optional[Mapping[str, Any]] = None,
    reproduce_script_src: Optional[Union[str, Path]] = None,
    figure_stem: str = STEM,
) -> Dict[str, Path]:
    """Persist PDF/PNG + all Haberman reproduction artifacts under *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_stem
    written: Dict[str, Path] = {}

    missing = [k for k in REQUIRED_GRID_KEYS if k not in grids]
    if missing:
        raise KeyError(f"Missing required grid arrays: {missing}")

    # --- Figure files ---------------------------------------------------------
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    written["pdf"] = pdf_path
    written["png"] = png_path

    # --- Points CSV --------------------------------------------------------
    required_cols = {"sample_index", "x_display", "y_display", "true_label"}
    if not required_cols.issubset(points_df.columns):
        raise ValueError(
            f"points_df missing columns {required_cols - set(points_df.columns)}"
        )
    points_path = output_dir / f"{stem}_points.csv"
    points_df.to_csv(points_path, index=False)
    written["points_csv"] = points_path

    # --- Grids NPZ ---------------------------------------------------------
    grids_path = output_dir / f"{stem}_grids.npz"
    np.savez_compressed(
        grids_path,
        **{k: np.asarray(v) for k, v in grids.items()},
    )
    written["grids_npz"] = grids_path

    # --- Plot config JSON --------------------------------------------------
    cfg_path = output_dir / f"{stem}_plot_config.json"
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(dict(plot_config), fh, indent=2, default=_json_default)
    written["plot_config"] = cfg_path

    # --- Checkpoint --------------------------------------------------------
    if model is not None:
        ckpt_path = output_dir / f"{stem}_checkpoint.pt"
        state = {
            "model_state_dict": cpu_state_dict(model),
            "model_config": dict(model_config or extract_model_config(model)),
            "model_class": (model_config or extract_model_config(model)).get(
                "model_class", model.__class__.__name__
            ),
            "model_module": (model_config or extract_model_config(model)).get(
                "model_module", model.__class__.__module__
            ),
        }
        for key in (
            "dataset",
            "seed",
            "run_index",
            "fold_index",
            "epoch",
            "number_of_rules",
            "feature_names",
            "feature_indices",
            "class_mapping",
            "rule_display_mapping",
            "metrics",
        ):
            if key in metadata and metadata[key] is not None:
                state[key] = metadata[key]
        if optimizer is not None and hasattr(optimizer, "state_dict"):
            try:
                state["optimizer_state_dict"] = {
                    k: (
                        {kk: vv.detach().cpu() if torch.is_tensor(vv) else vv
                         for kk, vv in v.items()}
                        if isinstance(v, dict)
                        else v
                    )
                    for k, v in optimizer.state_dict().items()
                }
            except Exception:
                # Optional — never block figure artifacts on optimizer dump.
                pass
        torch.save(state, ckpt_path)
        written["checkpoint"] = ckpt_path

    # --- Scaler / preprocessing --------------------------------------------
    if scaler is not None:
        scaler_path = output_dir / f"{stem}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        written["scaler"] = scaler_path
    if preprocessing_bundle is not None:
        bundle_path = output_dir / f"{stem}_preprocessing.joblib"
        joblib.dump(dict(preprocessing_bundle), bundle_path)
        written["preprocessing"] = bundle_path

    # --- Metadata JSON -----------------------------------------------------
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "package_versions": package_versions(),
        "artifact_dir": str(output_dir),
        "figure_stem": stem,
    }
    meta.update({k: v for k, v in dict(metadata).items() if v is not None})
    if "checkpoint" in written:
        meta["checkpoint_path"] = str(written["checkpoint"])
    if "scaler" in written:
        meta["scaler_path"] = str(written["scaler"])
    meta_path = output_dir / f"{stem}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=_json_default)
    written["metadata"] = meta_path

    # --- Bundle a standalone reproduce script ------------------------------
    if reproduce_script_src is not None:
        src = Path(reproduce_script_src)
        if src.is_file():
            dst = output_dir / "reproduce_haberman_boundary.py"
            shutil.copy2(src, dst)
            written["reproduce_script"] = dst

    return written
