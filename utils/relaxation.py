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
    cmap: str = "magma",
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
        cmap: colormap; the default (``magma``) maps low ``r`` -> dark.
        annotate: write the numeric ``r`` value inside each cell.
        figsize: figure size (auto-scaled from the matrix shape if None).
        ax: existing Axes to draw into (a new figure is made if None).
        title: custom plot title.
        save_path: if given, save the figure there.
    """
    r = relaxation_matrix(model)
    n_rules, n_features = r.shape

    if feature_names is None:
        feature_names = [f"F{j}" for j in range(n_features)]
    else:
        feature_names = list(feature_names[:n_features])
    if rule_names is None:
        rule_names = [f"Rule {i}" for i in range(n_rules)]

    created = ax is None
    if created:
        if figsize is None:
            figsize = (max(6, 0.6 * n_features + 2), max(3, 0.6 * n_rules + 1.5))
        _, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(r, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_rules))
    ax.set_yticklabels(rule_names, fontsize=9)
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Rule", fontsize=10)

    if annotate:
        # Choose readable text colour against the cell's brightness.
        rgba = im.cmap(im.norm(r))
        luminance = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
        for i in range(n_rules):
            for j in range(n_features):
                ax.text(
                    j, i, f"{r[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if luminance[i, j] > 0.5 else "white",
                )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("relaxation  r  (0 = active, 1 = ignored)", fontsize=9)

    ax.set_title(
        title or "Consequent relaxation heatmap (r per rule x feature)",
        fontsize=11, fontweight="bold",
    )

    if created:
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved relaxation heatmap to {save_path}")
    return ax
