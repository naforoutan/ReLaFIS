#!/usr/bin/env python3
"""Model-free reproduction of the HabermanBoundary paper figure.

Loads only:
  HabermanBoundary_points.csv
  HabermanBoundary_grids.npz
  HabermanBoundary_plot_config.json
  HabermanBoundary_metadata.json  (optional)

Does NOT import ReLaFIS/GIFTSHIFTER, load checkpoints, run inference,
retrain, recompute dominant rules, or refit scalers.

Usage:
  python scripts/reproduce_haberman_boundary.py \\
      --artifact-dir artifacts/figures/HabermanBoundary/seed_42_run_00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

# Allow running from repo root or from an artifact directory copy.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path and (_ROOT / "utils").is_dir():
    sys.path.insert(0, str(_ROOT))


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def _load_artifacts(artifact_dir: Path, stem: str = "HabermanBoundary"):
    points = pd.read_csv(_require(artifact_dir / f"{stem}_points.csv"))
    grids_path = _require(artifact_dir / f"{stem}_grids.npz")
    with np.load(grids_path) as data:
        grids = {k: data[k] for k in data.files}
    with _require(artifact_dir / f"{stem}_plot_config.json").open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    meta_path = artifact_dir / f"{stem}_metadata.json"
    meta = {}
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
    return points, grids, cfg, meta


def _haberman_style_axis(ax, tick_size: float) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4B5563")
        spine.set_linewidth(0.75)
    ax.tick_params(
        colors="#374151",
        labelsize=tick_size,
        width=0.75,
        length=3.0,
        direction="out",
        pad=2,
    )
    ax.grid(False)


def reproduce(artifact_dir: Path, *, stem: str = "HabermanBoundary") -> Path:
    points, grids, cfg, _meta = _load_artifacts(artifact_dir, stem=stem)

    required = [
        "xx", "yy",
        "dominant_rule_grid_display",
        "predicted_class_grid",
    ]
    for key in required:
        if key not in grids:
            # Fallbacks for older artifact naming.
            if key == "dominant_rule_grid_display" and "displayed_rule_grid" in grids:
                grids["dominant_rule_grid_display"] = grids["displayed_rule_grid"]
            elif key == "dominant_rule_grid_display" and "dominant_rule_grid" in grids:
                grids["dominant_rule_grid_display"] = grids["dominant_rule_grid"]
            else:
                raise KeyError(f"grids.npz missing '{key}'")

    xx = grids["xx"]
    yy = grids["yy"]
    rules = grids["dominant_rule_grid_display"]
    pred = grids["predicted_class_grid"]
    proba = grids.get("predicted_probability_grid")

    rule_colors = list(cfg["rule_colors"])
    class_fills = list(cfg["class_fill_colors"])
    point_colors = {int(k): v for k, v in cfg["point_colors"].items()}
    boundary_color = cfg["boundary_color"]
    figsize = tuple(cfg["figure_size"])
    fonts = cfg.get("font_sizes", {})
    title_size = fonts.get("panel_title", 10.5)
    axis_size = fonts.get("axis_label", 9.5)
    tick_size = fonts.get("tick", 8.5)
    legend_title = fonts.get("legend_title", 8.5)
    legend_size = fonts.get("legend", 8.0)
    marker_size = cfg.get("marker_size", 32)
    marker_alpha = cfg.get("marker_alpha", 0.98)
    marker_edge = cfg.get("marker_edgewidth", 0.5)
    boundary_lw = cfg.get("boundary_width", 0.8)
    titles = cfg.get("panel_titles", [
        "(a) Dominant rule regions",
        "(b) Predicted class regions",
    ])
    axis_labels = cfg.get("axis_labels", {})
    xlabel = axis_labels.get("x", "Positive axillary nodes (standardized)")
    ylabel = axis_labels.get("y", "Age (standardized)")
    xlim = cfg.get("x_limits")
    ylim = cfg.get("y_limits")
    class_names = cfg.get("legend_labels", {}).get("classes") or [
        r"Survived $\geq$ 5 years",
        "Died within 5 years",
    ]
    decision_level = float(cfg.get("contour_levels_decision", [0.5])[0])

    # Serif fonts from config when available.
    font_family = cfg.get("font_family") or ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": font_family,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    n_rules = len(rule_colors)
    n_classes = len(class_fills)
    rule_levels = np.arange(-0.5, n_rules + 0.5, 1)
    class_levels = np.arange(-0.5, n_classes + 0.5, 1)
    rule_cmap = ListedColormap(rule_colors)
    class_cmap = ListedColormap(class_fills)
    rule_norm = BoundaryNorm(rule_levels, rule_cmap.N)
    class_norm = BoundaryNorm(class_levels, class_cmap.N)

    fig, axes = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True, facecolor="white",
    )

    # Panel (a)
    ax0 = axes[0]
    _haberman_style_axis(ax0, tick_size)
    ax0.contourf(
        xx, yy, rules,
        levels=rule_levels, cmap=rule_cmap, norm=rule_norm,
        alpha=1.0, antialiased=True, corner_mask=True, zorder=0,
    )
    if n_rules > 1:
        ax0.contour(
            xx, yy, rules,
            levels=np.arange(0.5, n_rules - 0.5 + 1, 1),
            colors=boundary_color, linewidths=boundary_lw, alpha=0.95,
            antialiased=True, zorder=2,
        )
    ax0.set_title(titles[0], fontsize=title_size, fontweight="bold", pad=4)
    ax0.set_xlabel(xlabel, fontsize=axis_size, fontweight="bold", labelpad=3)
    ax0.set_ylabel(ylabel, fontsize=axis_size, fontweight="bold", labelpad=3)

    # Panel (b)
    ax1 = axes[1]
    _haberman_style_axis(ax1, tick_size)
    ax1.contourf(
        xx, yy, pred,
        levels=class_levels, cmap=class_cmap, norm=class_norm,
        alpha=1.0, antialiased=True, corner_mask=True, zorder=0,
    )
    if proba is not None:
        ax1.contour(
            xx, yy, proba, levels=[decision_level],
            colors=boundary_color, linewidths=boundary_lw, alpha=0.95,
            antialiased=True, zorder=2,
        )
    elif n_classes > 1:
        ax1.contour(
            xx, yy, pred,
            levels=np.arange(0.5, n_classes - 0.5 + 1, 1),
            colors=boundary_color, linewidths=boundary_lw, alpha=0.95,
            antialiased=True, zorder=2,
        )
    ax1.set_title(titles[1], fontsize=title_size, fontweight="bold", pad=4)
    ax1.set_xlabel(xlabel, fontsize=axis_size, fontweight="bold", labelpad=3)
    ax1.set_ylabel(ylabel, fontsize=axis_size, fontweight="bold", labelpad=3)

    # Scatter from exact saved coordinates
    x_pts = points["x_display"].to_numpy(dtype=float)
    y_pts = points["y_display"].to_numpy(dtype=float)
    y_true = points["true_label"].to_numpy()
    for ax, with_labels in ((axes[0], False), (axes[1], True)):
        for cls in np.unique(y_true):
            mask = y_true == cls
            color = point_colors.get(int(cls), "#000000")
            label = class_names[int(cls)] if int(cls) < len(class_names) else f"Class {int(cls)}"
            ax.scatter(
                x_pts[mask], y_pts[mask],
                s=marker_size, c=color, alpha=marker_alpha,
                edgecolors="white", linewidths=marker_edge,
                marker=cfg.get("marker", "o"), zorder=3,
                label=label if with_labels else None,
            )

    rule_handles = [
        Patch(facecolor=c, edgecolor="none", label=f"Rule {i + 1}")
        for i, c in enumerate(rule_colors)
    ]
    leg0 = ax0.legend(
        handles=rule_handles, title="Dominant rule", loc="upper right",
        fontsize=legend_size, title_fontsize=legend_title,
        frameon=True, framealpha=0.90, borderpad=0.25, labelspacing=0.20,
        handlelength=1.0, handletextpad=0.30, borderaxespad=0.35,
    )
    if leg0.get_title() is not None:
        leg0.get_title().set_fontweight("bold")

    leg1 = ax1.legend(
        title="True class", loc="upper right",
        fontsize=legend_size, title_fontsize=legend_title,
        frameon=True, framealpha=0.90, borderpad=0.25, labelspacing=0.20,
        handlelength=1.0, handletextpad=0.30, borderaxespad=0.35,
    )
    if leg1.get_title() is not None:
        leg1.get_title().set_fontweight("bold")

    if xlim is not None:
        ax0.set_xlim(xlim)
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax0.set_ylim(ylim)
        ax1.set_ylim(ylim)

    pdf_out = artifact_dir / f"{stem}_reproduced.pdf"
    png_out = artifact_dir / f"{stem}_reproduced.png"
    fig.savefig(pdf_out, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_out, format="png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Reproduced: {pdf_out}")
    print(f"Reproduced: {png_out}")
    return png_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing HabermanBoundary_* artifact files",
    )
    parser.add_argument("--stem", default="HabermanBoundary")
    args = parser.parse_args()
    reproduce(args.artifact_dir.resolve(), stem=args.stem)


if __name__ == "__main__":
    main()
