"""Relaxation heatmap for GIFT-family neuro-fuzzy models.

Unlike a decision boundary, the per-feature/per-rule consequent relaxation
coefficient ``r_{i,j}`` has no 2D limitation: it is a full ``rules x features``
matrix that can be shown directly as a heatmap.

For each rule ``i`` and feature ``j`` (following the model's ``tsk`` /
``relaxation_rate`` definition)::

    alpha1 = sigmoid(literal)      # Gaussian (equal / not-equal) branch commitment
    alpha2 = sigmoid(temp)         # relational (greater / less) branch commitment
    beta   = sigmoid(comb_weight)  # Gaussian vs relational mixing

    H1 = -alpha1 * log(alpha1) / (1/e)   in [0, 1]
    H2 = -alpha2 * log(alpha2) / (1/e)   in [0, 1]
    r  = beta * H1 + (1 - beta) * H2     in [0, 1]

``r`` near 0 means the feature is fully *active* in that rule (committed
antecedent, its consequent slope passes through un-attenuated). ``r`` near 1
means the feature is *relaxed / ignored* (uncommitted "don't care"). With the
default colormap dark cells are active and light cells are relaxed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.paper_plot_style import apply_paper_style
from utils.plot_style import ROSE_NAVY_SEQUENCE
from utils.verbalize_rules import clean_feature_name

# Colormap endpoints — do not alter; used for the Haberman relaxation figure.
_RELAXATION_CREAM = "#F5EDE4"
_HABERMAN_RELAX_FIGSIZE = (4.8, 3.35)
_PAPER_FEATURE_LABELS = {
    "age": "Age",
    "operation_year": "Operation year",
    "op. year": "Operation year",
    "positive_auxillary_nodes": "Nodes",
    "positive_axillary_nodes": "Nodes",
    "nodes": "Nodes",
}


def _relaxation_cmap() -> LinearSegmentedColormap:
    """Dark navy/plum (active) → dusty rose → warm cream (ignored)."""
    colors = list(reversed(ROSE_NAVY_SEQUENCE)) + [_RELAXATION_CREAM]
    return LinearSegmentedColormap.from_list("relaxation", colors)


def _paper_feature_label(name: str) -> str:
    """Publication tick labels for Haberman-style features."""
    base = name.split("__")[-1].lower().replace(" ", "_")
    if base in _PAPER_FEATURE_LABELS:
        return _PAPER_FEATURE_LABELS[base]
    if base.endswith("_age") or base == "age":
        return "Age"
    if "operation" in base and "year" in base:
        return "Operation year"
    if "node" in base or "auxillar" in base:
        return "Nodes"
    return clean_feature_name(name)


def _unwrap_model(model):
    """Return the underlying ``nn.Module`` whether given a model or a wrapper."""
    if hasattr(model, "literal") and hasattr(model, "comb_weight"):
        return model
    if hasattr(model, "model"):
        return model.model
    raise ValueError(
        "Expected a GIFT-family nn.Module (with .literal/.temp/"
        ".comb_weight) or an sklearn wrapper exposing .model"
    )


def relaxation_matrix(model) -> np.ndarray:
    """Compute the ``(rules, in_features)`` relaxation matrix ``r_{i,j}``."""
    model = _unwrap_model(model)
    if not (hasattr(model, "literal") and hasattr(model, "temp")
            and hasattr(model, "comb_weight")):
        raise ValueError(
            "Model does not expose the two-branch antecedent parameters "
            "(literal/temp/comb_weight) required for a relaxation heatmap."
        )

    eps = 1e-10
    one_over_e = 1.0 / np.e
    with torch.no_grad():
        alpha1 = torch.sigmoid(model.literal)       # (in_features, rules)
        alpha2 = torch.sigmoid(model.temp)          # (in_features, rules)
        beta = torch.sigmoid(model.comb_weight)     # (in_features, rules)

        H1 = -(alpha1 * torch.log(alpha1 + eps)) / one_over_e
        H2 = -(alpha2 * torch.log(alpha2 + eps)) / one_over_e
        r = beta * H1 + (1.0 - beta) * H2           # (in_features, rules)

    # Transpose to (rules, in_features) so rows are rules, columns are features.
    return r.T.cpu().numpy()


def _draw_relaxation_heatmap(
    r: np.ndarray,
    feature_names: List[str],
    rule_names: List[str],
    *,
    cmap=None,
    annotate: bool = True,
    figsize: Tuple[float, float] = _HABERMAN_RELAX_FIGSIZE,
    ax: Optional[plt.Axes] = None,
    title: str = "Consequent relaxation",
):
    """Render a compact publication heatmap from an already-computed matrix."""
    apply_paper_style()
    if cmap is None:
        cmap = _relaxation_cmap()

    r = np.asarray(r, dtype=np.float64)
    n_rules, n_features = r.shape

    created = ax is None
    if created:
        fig, ax = plt.subplots(
            figsize=figsize, constrained_layout=True, facecolor="white",
        )
    else:
        fig = ax.figure

    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#4B5563")

    im = ax.imshow(
        r,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )

    # Subtle cell separators.
    ax.set_xticks(np.arange(-0.5, n_features, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rules, 1), minor=True)
    ax.grid(which="minor", color="#D8D8D8", linestyle="-", linewidth=0.7, zorder=2)
    ax.tick_params(which="minor", size=0)

    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(
        feature_names,
        rotation=28,
        ha="right",
        rotation_mode="anchor",
        fontsize=8.5,
    )
    ax.set_yticks(np.arange(n_rules))
    ax.set_yticklabels(rule_names, rotation=0, va="center", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=8.5, length=0, pad=3)

    ax.set_xlabel("Feature", fontsize=9.5, fontweight="bold", labelpad=5)
    ax.set_ylabel("Rule", fontsize=9.5, fontweight="bold", labelpad=6)

    if annotate:
        rgba = im.cmap(im.norm(r))
        luminance = (
            0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
        )
        for i in range(n_rules):
            for j in range(n_features):
                # Light text on dark cells; dark text on light cells.
                text_color = "white" if luminance[i, j] < 0.52 else "#172033"
                ax.text(
                    j, i, f"{r[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=8.7, fontweight="normal",
                    color=text_color, zorder=3,
                )

    cbar = fig.colorbar(
        im, ax=ax, shrink=0.96, pad=0.035, aspect=24,
        ticks=np.linspace(0.0, 1.0, 6),
    )
    cbar.set_label(
        r"Relaxation $r$",
        fontsize=9, fontweight="bold", labelpad=5,
    )
    cbar.ax.tick_params(labelsize=8.2, width=0.7, length=3, pad=2)

    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)

    return fig, ax, im


def plot_relaxation_heatmap(
    model,
    feature_names: Optional[List[str]] = None,
    rule_names: Optional[List[str]] = None,
    cmap=None,
    annotate: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Draw the ``rules x features`` relaxation heatmap of ``r_{i,j}``.

    Dark cells => the feature is fully *active* in that rule (``r ~ 0``).
    Light cells => the feature is *relaxed / ignored* (``r ~ 1``).
    """
    if cmap is None:
        cmap = _relaxation_cmap()
    r = relaxation_matrix(model)
    n_rules, n_features = r.shape

    if feature_names is None:
        feature_names = [f"F{j}" for j in range(n_features)]
    else:
        feature_names = [_paper_feature_label(n) for n in feature_names[:n_features]]
    if rule_names is None:
        rule_names = [f"Rule {i + 1}" for i in range(n_rules)]

    if figsize is None:
        figsize = _HABERMAN_RELAX_FIGSIZE
    if title is None:
        title = "Consequent relaxation"

    fig, ax, _im = _draw_relaxation_heatmap(
        r, feature_names, rule_names,
        cmap=cmap, annotate=annotate, figsize=figsize, ax=ax, title=title,
    )

    if save_path:
        written = _save_relaxation_artifacts(
            fig=fig,
            r=r,
            feature_names=feature_names,
            rule_names=rule_names,
            figsize=figsize,
            title=title,
            save_path=save_path,
            cmap_name=getattr(cmap, "name", "relaxation"),
        )
        print(
            "Saved relaxation heatmap: "
            + ", ".join(str(p) for p in written.values())
        )
    return ax


def _save_relaxation_artifacts(
    *,
    fig,
    r: np.ndarray,
    feature_names: List[str],
    rule_names: List[str],
    figsize: Tuple[float, float],
    title: str,
    save_path: Union[str, Path],
    cmap_name: str,
):
    stem_path = Path(save_path)
    if stem_path.suffix:
        figure_dir = stem_path.parent
        stem = stem_path.stem
        # Notebook / script convention: .../HabermanRelaxation/HabermanRelaxation.pdf
        if figure_dir.name == "HabermanRelaxation" and stem != "HabermanRelaxation":
            stem = "HabermanRelaxation"
    else:
        # Prefer dedicated HabermanRelaxation collection directory.
        # Handle both:
        #   artifacts/figures/HabermanRelaxation
        #   artifacts/figures/HabermanRelaxation/HabermanRelaxation
        if stem_path.parent.name == "HabermanRelaxation":
            figure_dir = stem_path.parent
            stem = "HabermanRelaxation"
        elif stem_path.name == "HabermanRelaxation":
            figure_dir = stem_path
            stem = "HabermanRelaxation"
        else:
            figure_dir = stem_path.parent / "HabermanRelaxation"
            stem = "HabermanRelaxation"
    figure_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = figure_dir / f"{stem}.pdf"
    png_path = figure_dir / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight", pad_inches=0.03)

    # Wide CSV: Rule,Age,Operation year,Nodes
    values_df = pd.DataFrame(r, columns=feature_names)
    values_df.insert(0, "Rule", rule_names)
    values_csv = figure_dir / f"{stem}_values.csv"
    values_df.to_csv(values_csv, index=False)

    plot_config = {
        "figure_name": stem,
        "figure_size": list(figsize),
        "title": title,
        "feature_names": list(feature_names),
        "rule_names": list(rule_names),
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap_name": cmap_name,
        "cmap_sequence": list(reversed(ROSE_NAVY_SEQUENCE)) + [_RELAXATION_CREAM],
        "annotation_format": "{:.2f}",
        "annotation_fontsize": 8.7,
        "font_sizes": {
            "title": 10.5,
            "axis_label": 9.5,
            "tick": 8.5,
            "colorbar_label": 9.0,
            "colorbar_tick": 8.2,
        },
        "colorbar": {
            "label": r"Relaxation $r$",
            "ticks": list(np.linspace(0.0, 1.0, 6)),
            "shrink": 0.96,
            "pad": 0.035,
            "aspect": 24,
        },
        "grid": {"linewidth": 0.7, "linecolor": "#D8D8D8"},
        "square_cells": True,
        "output_filenames": {
            "pdf": f"{stem}.pdf",
            "png": f"{stem}.png",
            "values_csv": f"{stem}_values.csv",
        },
        "source_script": "utils/relaxation.py",
    }
    cfg_path = figure_dir / f"{stem}_plot_config.json"
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(plot_config, fh, indent=2)

    # Also keep NPZ for exact float arrays.
    np.savez_compressed(figure_dir / f"{stem}_grids.npz", relaxation=np.asarray(r))

    return {
        "pdf": pdf_path,
        "png": png_path,
        "values_csv": values_csv,
        "plot_config": cfg_path,
    }


def redraw_relaxation_from_artifacts(
    artifact_dir: Union[str, Path],
    *,
    stem: str = "HabermanRelaxation",
    save: bool = True,
):
    """Regenerate the heatmap from saved CSV/JSON only (no model)."""
    artifact_dir = Path(artifact_dir)
    values_csv = artifact_dir / f"{stem}_values.csv"
    cfg_path = artifact_dir / f"{stem}_plot_config.json"
    if not values_csv.is_file():
        raise FileNotFoundError(values_csv)
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)

    df = pd.read_csv(values_csv)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    rule_names = df["Rule"].tolist()
    feature_names = [c for c in df.columns if c != "Rule"]
    r = df[feature_names].to_numpy(dtype=np.float64)

    # Rebuild the exact same colormap sequence from saved colors.
    colors = cfg.get("cmap_sequence")
    if colors:
        cmap = LinearSegmentedColormap.from_list("relaxation", colors)
    else:
        cmap = _relaxation_cmap()

    figsize = tuple(cfg.get("figure_size", _HABERMAN_RELAX_FIGSIZE))
    title = cfg.get("title", "Consequent relaxation")
    fig, ax, _ = _draw_relaxation_heatmap(
        r, feature_names, rule_names,
        cmap=cmap, annotate=True, figsize=figsize, title=title,
    )
    if save:
        pdf = artifact_dir / f"{stem}_reproduced.pdf"
        png = artifact_dir / f"{stem}_reproduced.png"
        fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(png, format="png", dpi=600, bbox_inches="tight", pad_inches=0.03)
        print(f"Reproduced: {pdf}")
        print(f"Reproduced: {png}")
    return fig, ax
