"""2D decision-boundary and rule-region visualization for neuro-fuzzy models.

These helpers demonstrate *interpretability*: they show how the learned fuzzy
rules partition the input space and how that partition induces the model's
decision boundary.

Because a decision boundary can only be drawn in 2D, the input space is
projected onto two chosen features (e.g. Haberman's "positive axillary nodes"
vs "age"). Every other feature is held fixed at a constant value (its mean by
default, which is ~0 for the standardized data these models are trained on).
A dense grid is then swept over the two free features and, for each grid point,
we ask the model two questions:

    * which rule fires most strongly there  -> ``plot_rule_regions_2d``
    * which class does the model predict     -> ``plot_decision_boundary_2d``

Colouring the grid by the dominant rule gives visual proof that the rules
genuinely carve the input space into contiguous regions, and overlaying the
predicted-class boundary shows how those regions combine into a decision.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _unwrap_model(model):
    """Return the underlying ``nn.Module`` whether given a model or a wrapper."""
    if hasattr(model, "encode") and hasattr(model, "rules_count"):
        return model
    if hasattr(model, "model"):
        return model.model
    raise ValueError(
        "Expected a neuro-fuzzy nn.Module (with .encode/.rules_count) or an "
        "sklearn wrapper exposing .model"
    )


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _build_projected_grid(
    X: np.ndarray,
    feature_x: int,
    feature_y: int,
    fixed_values: Optional[Sequence[float]] = None,
    resolution: int = 300,
    padding: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dense grid over two features, holding the rest fixed.

    Returns ``(xx, yy, grid)`` where ``xx``/``yy`` are the 2D meshgrid arrays
    for the two free features and ``grid`` is the ``(resolution**2, n_features)``
    array of full-dimensional inputs to feed the model.
    """
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]

    if fixed_values is None:
        fixed_values = X.mean(axis=0)
    fixed_values = np.asarray(fixed_values, dtype=np.float64)

    x_min, x_max = X[:, feature_x].min(), X[:, feature_x].max()
    y_min, y_max = X[:, feature_y].min(), X[:, feature_y].max()
    x_pad = (x_max - x_min) * padding + 1e-6
    y_pad = (y_max - y_min) * padding + 1e-6

    xs = np.linspace(x_min - x_pad, x_max + x_pad, resolution)
    ys = np.linspace(y_min - y_pad, y_max + y_pad, resolution)
    xx, yy = np.meshgrid(xs, ys)

    # Start every grid point from the fixed background, then vary the 2 axes.
    grid = np.tile(fixed_values, (xx.size, 1))
    grid[:, feature_x] = xx.ravel()
    grid[:, feature_y] = yy.ravel()

    return xx, yy, grid


def _dominant_rule(model, grid: np.ndarray) -> np.ndarray:
    """Return the index of the strongest-firing rule at each grid point."""
    model = _unwrap_model(model)
    model.eval()
    device = _model_device(model)
    with torch.no_grad():
        X = torch.as_tensor(grid, dtype=torch.float32, device=device)
        firing = model.encode(X)
        if model.rules_count > 1:
            firing = F.normalize(firing, p=1, dim=1)
        rules = firing.argmax(dim=1).cpu().numpy()
    return rules


def _predict_grid(model, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(predicted_class, confidence)`` at each grid point."""
    model = _unwrap_model(model)
    model.eval()
    device = _model_device(model)
    with torch.no_grad():
        X = torch.as_tensor(grid, dtype=torch.float32, device=device)
        logits = model(X)
        if isinstance(logits, tuple):
            logits = logits[0]
        if getattr(model, "binary", False) or logits.shape[1] == 1:
            proba = torch.sigmoid(logits).squeeze(-1)
            pred = (proba > 0.5).long()
            conf = torch.where(pred.bool(), proba, 1 - proba)
        else:
            proba = torch.softmax(logits, dim=1)
            conf, pred = proba.max(dim=1)
    return pred.cpu().numpy(), conf.cpu().numpy()


def plot_rule_regions_2d(
    model,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    feature_x: int = 0,
    feature_y: int = 1,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    fixed_values: Optional[Sequence[float]] = None,
    resolution: int = 300,
    ax: Optional[plt.Axes] = None,
    scatter: bool = True,
    cmap: str = "tab10",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Colour the 2D input plane by which rule dominates in each region.

    Args:
        model: trained neuro-fuzzy model (or its sklearn wrapper).
        X: reference data, shape ``(n_samples, n_features)`` — used to set the
            plot range and the fixed background values.
        y: optional labels for the overlaid scatter of the real data.
        feature_x, feature_y: indices of the two features to vary.
        feature_names: names for axis labels (indexed by feature position).
        class_names: names for the scatter legend.
        fixed_values: values for the held-fixed features (default: ``X.mean(0)``).
        resolution: grid resolution per axis.
        ax: existing matplotlib Axes to draw into (a new figure is made if None).
        scatter: whether to overlay the real data points.
        cmap: colormap for the rule regions.
        title: custom plot title.
        save_path: if given, save the figure there.
    """
    model_u = _unwrap_model(model)
    n_rules = model_u.rules_count

    xx, yy, grid = _build_projected_grid(
        X, feature_x, feature_y, fixed_values, resolution
    )
    rules = _dominant_rule(model, grid).reshape(xx.shape)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7, 6))

    base = plt.get_cmap(cmap)
    rule_cmap = ListedColormap([base(i % base.N) for i in range(n_rules)])

    ax.contourf(
        xx, yy, rules,
        levels=np.arange(-0.5, n_rules + 0.5, 1),
        cmap=rule_cmap, alpha=0.45,
    )
    # Draw crisp borders between rule regions.
    ax.contour(
        xx, yy, rules,
        levels=np.arange(0.5, n_rules - 0.5 + 1, 1),
        colors="k", linewidths=1.0, alpha=0.6,
    )

    _overlay_scatter(ax, X, y, feature_x, feature_y, class_names, scatter)
    _label_axes(ax, feature_x, feature_y, feature_names)

    # Legend entries for the rule regions.
    rule_handles = [
        plt.Rectangle((0, 0), 1, 1, color=rule_cmap(i), alpha=0.6,
                      label=f"Rule {i}")
        for i in range(n_rules)
    ]
    region_legend = ax.legend(
        handles=rule_handles, title="Dominant rule",
        loc="upper right", fontsize=8, framealpha=0.9,
    )
    ax.add_artist(region_legend)

    ax.set_title(title or "Rule regions (dominant rule per area)",
                 fontsize=11, fontweight="bold")

    if created:
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_decision_boundary_2d(
    model,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    feature_x: int = 0,
    feature_y: int = 1,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    fixed_values: Optional[Sequence[float]] = None,
    resolution: int = 300,
    ax: Optional[plt.Axes] = None,
    scatter: bool = True,
    cmap: str = "coolwarm",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Colour the 2D input plane by the model's predicted class.

    See :func:`plot_rule_regions_2d` for the shared arguments.
    """
    xx, yy, grid = _build_projected_grid(
        X, feature_x, feature_y, fixed_values, resolution
    )
    pred, _ = _predict_grid(model, grid)
    pred = pred.reshape(xx.shape)
    n_classes = int(pred.max()) + 1
    n_classes = max(n_classes, len(np.unique(pred)))

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7, 6))

    levels = np.arange(-0.5, max(n_classes, 2) + 0.5, 1)
    ax.contourf(xx, yy, pred, levels=levels, cmap=cmap, alpha=0.4)
    # The decision boundary itself.
    ax.contour(xx, yy, pred, levels=np.arange(0.5, max(n_classes, 2) - 0.5 + 1, 1),
               colors="k", linewidths=1.6)

    _overlay_scatter(ax, X, y, feature_x, feature_y, class_names, scatter)
    _label_axes(ax, feature_x, feature_y, feature_names)

    ax.set_title(title or "Decision boundary", fontsize=11, fontweight="bold")

    if created:
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_interpretability_2d(
    model,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    feature_x: int = 0,
    feature_y: int = 1,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    fixed_values: Optional[Sequence[float]] = None,
    resolution: int = 300,
    figsize: Tuple[int, int] = (14, 6),
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Side-by-side rule regions and decision boundary on the same projection.

    This is the main entry point for demonstrating interpretability: the left
    panel proves the rules partition the input space, and the right panel shows
    the resulting decision boundary over the same two features.

    Args:
        model: trained neuro-fuzzy model (or its sklearn wrapper).
        X: reference data, shape ``(n_samples, n_features)``.
        y: optional labels for the overlaid scatter.
        feature_x, feature_y: indices of the two features to vary (e.g. the
            indices of "positive axillary nodes" and "age" for Haberman).
        feature_names: names for axis labels.
        class_names: names for the scatter/legend.
        fixed_values: held-fixed feature values (default: ``X.mean(0)``).
        resolution: grid resolution per axis.
        figsize: figure size.
        suptitle: overall figure title.
        save_path: if given, save the figure there.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_rule_regions_2d(
        model, X, y, feature_x, feature_y, feature_names, class_names,
        fixed_values, resolution, ax=axes[0],
    )
    plot_decision_boundary_2d(
        model, X, y, feature_x, feature_y, feature_names, class_names,
        fixed_values, resolution, ax=axes[1],
    )

    fig.suptitle(
        suptitle or "Rule partition vs. decision boundary",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved interpretability figure to {save_path}")
    else:
        plt.show()
    return fig, axes


def _overlay_scatter(ax, X, y, feature_x, feature_y, class_names, scatter):
    if not scatter:
        return
    X = np.asarray(X)
    xs, ys = X[:, feature_x], X[:, feature_y]
    if y is None:
        ax.scatter(xs, ys, c="k", s=14, alpha=0.5, edgecolors="white",
                   linewidths=0.4)
        return
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    palette = plt.get_cmap("Set1")
    for i, cls in enumerate(classes):
        mask = y == cls
        label = (class_names[int(cls)] if class_names is not None
                 and int(cls) < len(class_names) else f"Class {int(cls)}")
        ax.scatter(
            xs[mask], ys[mask], color=palette(i), s=22, alpha=0.9,
            edgecolors="white", linewidths=0.5, label=label,
        )
    ax.legend(title="True class", loc="upper left", fontsize=8, framealpha=0.9)


def _label_axes(ax, feature_x, feature_y, feature_names):
    if feature_names is not None:
        xlabel = feature_names[feature_x] if feature_x < len(feature_names) else f"x{feature_x}"
        ylabel = feature_names[feature_y] if feature_y < len(feature_names) else f"x{feature_y}"
    else:
        xlabel, ylabel = f"Feature {feature_x}", f"Feature {feature_y}"
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
