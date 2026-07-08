"""Relaxation heatmap for GIFTSHIFTER-family neuro-fuzzy models.

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

from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.plot_style import (
    ROSE_NAVY_SEQUENCE,
    apply_plot_style,
    panel_title,
    save_figure,
    style_axes_minimal,
)
from utils.verbalize_rules import clean_feature_name

_RELAXATION_CREAM = "#F5EDE4"
_SHORT_FEATURE_LABELS = {
    "age": "age",
    "operation_year": "op. year",
    "positive_auxillary_nodes": "nodes",
    "positive_axillary_nodes": "nodes",
}


def _relaxation_cmap() -> LinearSegmentedColormap:
    """Dark navy/plum (active) → dusty rose → warm cream (ignored)."""
    colors = list(reversed(ROSE_NAVY_SEQUENCE)) + [_RELAXATION_CREAM]
    return LinearSegmentedColormap.from_list("relaxation", colors)


def _short_feature_label(name: str) -> str:
    """Compact column labels for heatmaps."""
    base = name.split("__")[-1].lower().replace(" ", "_")
    if base in _SHORT_FEATURE_LABELS:
        return _SHORT_FEATURE_LABELS[base]
    if base.endswith("_age") or base == "age":
        return "age"
    if "operation" in base and "year" in base:
        return "op. year"
    if "node" in base or "auxillar" in base:
        return "nodes"
    return clean_feature_name(name)


def _unwrap_model(model):
    """Return the underlying ``nn.Module`` whether given a model or a wrapper."""
    if hasattr(model, "literal") and hasattr(model, "comb_weight"):
        return model
    if hasattr(model, "model"):
        return model.model
    raise ValueError(
        "Expected a GIFTSHIFTER-family nn.Module (with .literal/.temp/"
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


def plot_relaxation_heatmap(
    model,
    feature_names: Optional[List[str]] = None,
    rule_names: Optional[List[str]] = None,
    cmap=None,
    annotate: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Draw the ``rules x features`` relaxation heatmap of ``r_{i,j}``.

    Dark cells => the feature is fully *active* in that rule (``r ~ 0``).
    Light cells => the feature is *relaxed / ignored* (``r ~ 1``).

    Args:
        model: trained GIFTSHIFTER-family model (or its sklearn wrapper).
        feature_names: column labels (indexed by feature position).
        rule_names: row labels (one per rule).
        cmap: colormap; default maps low ``r`` (active) → dark plum/navy.
        annotate: write the numeric ``r`` value inside each cell.
        figsize: figure size (auto-scaled from the matrix shape if None).
        ax: existing Axes to draw into (a new figure is made if None).
        title: custom plot title.
        save_path: if given, save the figure there.
    """
    apply_plot_style()
    if cmap is None:
        cmap = _relaxation_cmap()
    r = relaxation_matrix(model)
    n_rules, n_features = r.shape

    if feature_names is None:
        feature_names = [f"F{j}" for j in range(n_features)]
    else:
        feature_names = [_short_feature_label(n) for n in feature_names[:n_features]]
    if rule_names is None:
        rule_names = [f"Rule {i + 1}" for i in range(n_rules)]

    created = ax is None
    if created:
        if figsize is None:
            figsize = (
                max(6.5, 0.55 * n_features + 1.8),
                max(5.2, 1.35 * n_rules + 2.8),
            )
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig = ax.figure

    style_axes_minimal(ax)
    im = ax.imshow(
        r, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(feature_names, rotation=28, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_rules))
    ax.set_yticklabels(rule_names, fontsize=9)
    ax.set_xlabel("Feature", labelpad=10)
    ax.set_ylabel("Rule", labelpad=10)

    ax.set_xticks(np.arange(-0.5, n_features, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rules, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6, zorder=2)
    ax.tick_params(which="minor", size=0)

    if annotate:
        rgba = im.cmap(im.norm(r))
        luminance = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
        for i in range(n_rules):
            for j in range(n_features):
                ax.text(
                    j, i, f"{r[i, j]:.2f}",
                    ha="center", va="center", fontsize=8.5, fontweight="600",
                    color="#1A202C" if luminance[i, j] > 0.52 else "white",
                    zorder=3,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    cbar.set_label("Relaxation r", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.annotate(
        "ignored", xy=(0.5, 1.03), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=7.5, color="#64748B",
    )
    cbar.ax.annotate(
        "active", xy=(0.5, -0.09), xycoords="axes fraction",
        ha="center", va="top", fontsize=7.5, color="#64748B",
    )

    panel_title(
        ax,
        title or "Haberman — consequent relaxation by rule and feature",
    )

    if created:
        fig.tight_layout()
    if save_path:
        paths = save_figure(fig, save_path)
        print(f"Saved relaxation heatmap: {', '.join(paths)}")
    return ax
