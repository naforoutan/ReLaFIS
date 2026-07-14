"""Publication Matplotlib style and figure-artifact helpers.

Typography hierarchy (Information Sciences–style proportions):

* Panel titles / ``(a)`` labels ≈ 11 pt bold
* Axis labels ≈ 10.5 pt bold
* Tick labels ≈ 9 pt
* Legends / annotations ≈ 8.5–9 pt

Do not force every text element to a large bold size.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.2

TITLE_SIZE = 11
AXIS_LABEL_SIZE = 10.5
TICK_SIZE = 9
LEGEND_SIZE = 8.5
ANNOTATION_SIZE = 8.5
CBAR_LABEL_SIZE = 9.5
CBAR_TICK_SIZE = 8.5

PAPER_RCPARAMS: Dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9.5,
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.labelweight": "bold",
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def apply_paper_style() -> None:
    """Apply balanced serif typography and save settings for paper figures."""
    mpl.rcParams.update(PAPER_RCPARAMS)


def style_axis(ax: plt.Axes, *, grid: bool = False) -> None:
    """Clean spines / ticks; optional subtle grid."""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4B5563")
        spine.set_linewidth(0.9)
    ax.tick_params(
        colors="#374151",
        labelsize=TICK_SIZE,
        width=0.9,
        length=3.5,
        direction="out",
        pad=3,
    )
    if grid:
        ax.grid(True, linewidth=0.6, alpha=0.35, color="#CBD5E1")
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def make_legend_compact(legend) -> None:
    """Tighten legend frame and text without oversized bold labels."""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#D1D5DB")
    frame.set_alpha(0.90)
    frame.set_linewidth(0.7)
    title = legend.get_title()
    if title is not None:
        title.set_fontsize(LEGEND_SIZE)
        title.set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontsize(LEGEND_SIZE)
        text.set_fontweight("normal")


def panel_label(ax: plt.Axes, text: str, *, pad: float = 5) -> None:
    """Short ``(a)`` / ``(b)`` panel title at ~11 pt bold."""
    ax.set_title(text, fontsize=TITLE_SIZE, fontweight="bold", pad=pad)


def save_paper_figure(
    fig: plt.Figure,
    output_path: Union[str, Path],
) -> List[str]:
    """Write vector PDF and 600-dpi PNG; create parent directories as needed."""
    apply_paper_style()
    base = Path(output_path)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    facecolor = fig.get_facecolor()
    for fmt in ("pdf", "png"):
        out = base.with_suffix(f".{fmt}")
        fig.savefig(
            out,
            dpi=600 if fmt == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor=facecolor,
        )
        saved.append(str(out))
    return saved


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
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


def save_figure_artifacts(
    figure_dir: Union[str, Path],
    *,
    figure_stem: Optional[str] = None,
    points_df: Optional[pd.DataFrame] = None,
    grid_dict: Optional[Mapping[str, np.ndarray]] = None,
    plot_config: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Persist CSV / NPZ / JSON artifacts needed to redraw a paper figure."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_stem or figure_dir.name
    written: Dict[str, str] = {}

    if points_df is not None:
        path = figure_dir / f"{stem}_points.csv"
        points_df.to_csv(path, index=False)
        written["points_csv"] = str(path)

    if grid_dict is not None:
        path = figure_dir / f"{stem}_grids.npz"
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in grid_dict.items()})
        written["grids_npz"] = str(path)

    if plot_config is not None:
        path = figure_dir / f"{stem}_plot_config.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(dict(plot_config), fh, indent=2, default=_json_default)
        written["plot_config"] = str(path)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
    }
    if metadata:
        meta.update(dict(metadata))
    path = figure_dir / f"{stem}_metadata.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=_json_default)
    written["metadata"] = str(path)
    return written


def load_figure_artifacts(
    figure_dir: Union[str, Path],
    *,
    figure_stem: Optional[str] = None,
) -> Dict[str, Any]:
    """Load previously saved figure artifacts from *figure_dir*."""
    figure_dir = Path(figure_dir)
    stem = figure_stem or figure_dir.name
    out: Dict[str, Any] = {"figure_dir": figure_dir, "figure_stem": stem}

    points = figure_dir / f"{stem}_points.csv"
    if points.exists():
        out["points_df"] = pd.read_csv(points)

    grids = figure_dir / f"{stem}_grids.npz"
    if grids.exists():
        with np.load(grids) as data:
            out["grids"] = {k: data[k] for k in data.files}

    cfg = figure_dir / f"{stem}_plot_config.json"
    if cfg.exists():
        with cfg.open(encoding="utf-8") as fh:
            out["plot_config"] = json.load(fh)

    meta = figure_dir / f"{stem}_metadata.json"
    if meta.exists():
        with meta.open(encoding="utf-8") as fh:
            out["metadata"] = json.load(fh)

    return out
