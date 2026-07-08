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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from utils.plot_style import (
    CLASS_SCATTER,
    ROSE_NAVY_SEQUENCE,
    class_cmap,
    panel_title,
    rule_cmap,
    save_figure,
    style_axes,
    style_axes_minimal,
    style_legend,
)
from utils.verbalize_rules import clean_feature_name

# Region backgrounds only — scatter styling is unchanged below.
_INTERP_REGION_ALPHA = 0.32
_INTERP_RULE_REGION_FILL = [
    "#D8A4B0",  # soft rose
    "#C4B0D4",  # pale lavender
    "#DDB8A8",  # soft peach
    "#A8C4D8",  # mist blue
]
_INTERP_CLASS_REGION_FILL = [
    "#F5DDE3",  # soft rose
    "#E4EFF6",  # mist blue
    "#FAEEE6",  # soft peach
    "#EDE8F4",  # pale lavender
]
_INTERP_RULE_BOUNDARY_COLOR = "white"
_INTERP_RULE_BOUNDARY_LW = 1.15
_INTERP_RULE_BOUNDARY_ALPHA = 1.0
_INTERP_RULE_BOUNDARY_EDGE_COLOR = "#A8889C"
_INTERP_RULE_BOUNDARY_EDGE_LW = 0.55
_INTERP_RULE_BOUNDARY_EDGE_ALPHA = 0.78
_INTERP_DECISION_BOUNDARY_COLOR = "#5A365F"  # muted plum/navy
_INTERP_DECISION_BOUNDARY_LW = 0.52
_INTERP_DECISION_BOUNDARY_ALPHA = 0.55
_INTERP_GRID_RESOLUTION = 475
_INTERP_SCATTER_ALPHA = 0.70
_INTERP_SCATTER_SIZE = 72
_INTERP_AXIS_MARGIN = 0.12


def _rule_cmap(n_rules: int) -> ListedColormap:
    return rule_cmap(n_rules)


def _class_cmap(n_classes: int) -> ListedColormap:
    return class_cmap(n_classes)


def _interp_rule_region_cmap(n: int) -> ListedColormap:
    """Pale pastels for dominant-rule ``contourf`` fills."""
    return ListedColormap(
        [_INTERP_RULE_REGION_FILL[i % len(_INTERP_RULE_REGION_FILL)] for i in range(max(n, 1))]
    )


def _interp_class_region_cmap(n: int) -> ListedColormap:
    """Pale pastels for predicted-class ``contourf`` fills."""
    return ListedColormap(
        [_INTERP_CLASS_REGION_FILL[i % len(_INTERP_CLASS_REGION_FILL)] for i in range(max(n, 1))]
    )


def _interp_scatter_cmap() -> LinearSegmentedColormap:
    """Deep plum → purple → navy for scatter bubbles."""
    return LinearSegmentedColormap.from_list(
        "interp_scatter", ROSE_NAVY_SEQUENCE[2:],
    )


def _set_interp_axis_limits(
    ax: plt.Axes, X: np.ndarray, feature_x: int, feature_y: int,
) -> None:
    """Tighten framing around the data cloud."""
    xs = np.asarray(X)[:, feature_x]
    ys = np.asarray(X)[:, feature_y]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_pad = (x_max - x_min) * _INTERP_AXIS_MARGIN + 1e-6
    y_pad = (y_max - y_min) * _INTERP_AXIS_MARGIN + 1e-6
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def _apply_panel_style(ax: plt.Axes, *, interpretability: bool = False) -> None:
    if interpretability:
        style_axes_minimal(ax)
    else:
        style_axes(ax)


def _style_legend(legend) -> None:
    style_legend(legend)


def _panel_title(ax: plt.Axes, title: str) -> None:
    panel_title(ax, title)


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
    interpretability: bool = False,
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

    _apply_panel_style(ax, interpretability=interpretability)
    if interpretability:
        region_cmap = _interp_rule_region_cmap(n_rules)
        region_alpha = _INTERP_REGION_ALPHA
    else:
        region_cmap = _rule_cmap(n_rules)
        region_alpha = 0.62

    ax.contourf(
        xx, yy, rules,
        levels=np.arange(-0.5, n_rules + 0.5, 1),
        cmap=region_cmap, alpha=region_alpha, antialiased=True, zorder=1,
    )
    if n_rules > 1:
        boundary_levels = np.arange(0.5, n_rules - 0.5 + 1, 1)
        if interpretability:
            ax.contour(
                xx, yy, rules, levels=boundary_levels,
                colors=_INTERP_RULE_BOUNDARY_COLOR, linewidths=_INTERP_RULE_BOUNDARY_LW,
                alpha=_INTERP_RULE_BOUNDARY_ALPHA, zorder=3,
            )
            ax.contour(
                xx, yy, rules, levels=boundary_levels,
                colors=_INTERP_RULE_BOUNDARY_EDGE_COLOR,
                linewidths=_INTERP_RULE_BOUNDARY_EDGE_LW,
                alpha=_INTERP_RULE_BOUNDARY_EDGE_ALPHA, zorder=3,
            )
        else:
            ax.contour(
                xx, yy, rules, levels=boundary_levels,
                colors="white", linewidths=2.0, alpha=0.85, zorder=2,
            )
            ax.contour(
                xx, yy, rules, levels=boundary_levels,
                colors="#2D3748", linewidths=0.8, alpha=0.45, zorder=2,
            )

    _overlay_scatter(
        ax, X, y, feature_x, feature_y, class_names, scatter,
        interpretability=interpretability,
        class_legend_y=0.58 if interpretability else 1.0,
    )
    _label_axes(ax, feature_x, feature_y, feature_names, interpretability=interpretability)
    if interpretability:
        _set_interp_axis_limits(ax, X, feature_x, feature_y)

    rule_handles = []
    for i in range(n_rules):
        if interpretability:
            rule_handles.append(Line2D(
                [], [], linestyle="none", marker="s", markersize=7,
                markerfacecolor=region_cmap(i), markeredgecolor="white",
                markeredgewidth=0.55, alpha=0.9, label=f"Rule {i + 1}",
            ))
        else:
            rule_handles.append(plt.Rectangle(
                (0, 0), 1, 1, facecolor=region_cmap(i), edgecolor="white",
                linewidth=0.6, label=f"Rule {i + 1}",
            ))
    rule_anchor = (1.0, 1.0) if interpretability else None
    region_legend = ax.legend(
        handles=rule_handles, title="Dominant rule",
        loc="upper right",
        bbox_to_anchor=rule_anchor,
        fontsize=8, framealpha=0.96, title_fontsize=8,
        labelspacing=0.65 if interpretability else None,
    )
    _style_legend(region_legend)
    ax.add_artist(region_legend)

    _panel_title(ax, title or "Rule regions (dominant rule per area)")

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
    interpretability: bool = False,
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

    _apply_panel_style(ax, interpretability=interpretability)
    levels = np.arange(-0.5, max(n_classes, 2) + 0.5, 1)
    if interpretability:
        region_cmap = _interp_class_region_cmap(max(n_classes, 2))
        region_alpha = _INTERP_REGION_ALPHA
    else:
        region_cmap = _class_cmap(n_classes)
        region_alpha = 0.72

    ax.contourf(
        xx, yy, pred, levels=levels, cmap=region_cmap,
        alpha=region_alpha, antialiased=True, zorder=1,
    )
    boundary_levels = np.arange(0.5, max(n_classes, 2) - 0.5 + 1, 1)
    if len(boundary_levels) > 0:
        if interpretability:
            ax.contour(
                xx, yy, pred, levels=boundary_levels,
                colors=_INTERP_DECISION_BOUNDARY_COLOR,
                linewidths=_INTERP_DECISION_BOUNDARY_LW,
                alpha=_INTERP_DECISION_BOUNDARY_ALPHA, zorder=2,
            )
        else:
            ax.contour(
                xx, yy, pred, levels=boundary_levels,
                colors="white", linewidths=2.2, alpha=0.9, zorder=2,
            )
            ax.contour(
                xx, yy, pred, levels=boundary_levels,
                colors="#1A202C", linewidths=1.1, alpha=0.7, zorder=2,
            )

    _overlay_scatter(
        ax, X, y, feature_x, feature_y, class_names, scatter,
        interpretability=interpretability,
    )
    _label_axes(ax, feature_x, feature_y, feature_names, interpretability=interpretability)
    if interpretability:
        _set_interp_axis_limits(ax, X, feature_x, feature_y)

    _panel_title(ax, title or "Decision boundary")

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
    resolution: int = _INTERP_GRID_RESOLUTION,
    figsize: Tuple[int, int] = (14, 6.5),
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
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True, facecolor="white",
    )

    grid_resolution = max(resolution, _INTERP_GRID_RESOLUTION)

    plot_rule_regions_2d(
        model, X, y, feature_x, feature_y, feature_names, class_names,
        fixed_values, grid_resolution, ax=axes[0],
        title="(A) Dominant rule regions",
        interpretability=True,
    )
    plot_decision_boundary_2d(
        model, X, y, feature_x, feature_y, feature_names, class_names,
        fixed_values, grid_resolution, ax=axes[1],
        title="(B) Predicted class regions",
        interpretability=True,
    )

    fig.suptitle(
        suptitle or "Rule partition vs. decision boundary",
        fontsize=13, fontweight="600", color="#0F172A", y=1.02,
    )
    fig.supxlabel(
        "Remaining features held at their mean value (standardized ≈ 0).",
        fontsize=8.5, color="#64748B", style="italic",
    )

    if save_path:
        paths = save_figure(fig, save_path)
        print(f"Saved interpretability figure: {', '.join(paths)}")
    else:
        plt.show()
    return fig, axes


def _overlay_scatter(
    ax, X, y, feature_x, feature_y, class_names, scatter, *,
    interpretability: bool = False,
    class_legend_y: float = 1.0,
):
    if not scatter:
        return
    X = np.asarray(X)
    xs, ys = X[:, feature_x], X[:, feature_y]
    if y is None:
        scatter_c = "#4B5563" if interpretability else "#2D3748"
        ax.scatter(
            xs, ys, c=scatter_c, s=_INTERP_SCATTER_SIZE if interpretability else 28,
            alpha=_INTERP_SCATTER_ALPHA if interpretability else 0.55,
            edgecolors="white", linewidths=0.65, zorder=4,
        )
        return
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    if interpretability:
        cmap = _interp_scatter_cmap()
        norm = Normalize(vmin=float(classes.min()), vmax=float(classes.max()))
        ax.scatter(
            xs, ys, c=y.astype(float), cmap=cmap, norm=norm,
            s=_INTERP_SCATTER_SIZE, alpha=_INTERP_SCATTER_ALPHA,
            edgecolors="white", linewidths=0.8, zorder=4,
        )
        _compact_class_legend(
            ax, classes, class_names, cmap, norm, anchor_y=class_legend_y,
        )
        return

    for i, cls in enumerate(classes):
        mask = y == cls
        label = (class_names[int(cls)] if class_names is not None
                 and int(cls) < len(class_names) else f"Class {int(cls)}")
        color = CLASS_SCATTER[i % len(CLASS_SCATTER)]
        ax.scatter(
            xs[mask], ys[mask], color=color, s=42, alpha=0.92,
            edgecolors="white", linewidths=0.9, label=label, zorder=4,
        )
    legend = ax.legend(
        title="True class", loc="upper left", fontsize=8, framealpha=0.96,
    )
    _style_legend(legend)


def _compact_class_legend(ax, classes, class_names, cmap, norm, *, anchor_y: float):
    """Compact top-right legend for dark scatter bubbles."""
    handles = []
    labels = []
    for cls in classes:
        label = (class_names[int(cls)] if class_names is not None
                 and int(cls) < len(class_names) else f"Class {int(cls)}")
        handles.append(Line2D(
            [], [], linestyle="none", marker="o", markersize=8.5,
            markerfacecolor=cmap(norm(float(cls))), markeredgecolor="white",
            markeredgewidth=0.7, alpha=0.85,
        ))
        labels.append(label)
    legend = ax.legend(
        handles, labels, title="True class",
        loc="upper right", bbox_to_anchor=(1.0, anchor_y),
        fontsize=8, title_fontsize=8, framealpha=0.96, labelspacing=0.65,
    )
    _style_legend(legend)


def _label_axes(ax, feature_x, feature_y, feature_names, *, interpretability: bool = False):
    if feature_names is not None:
        xlabel = (clean_feature_name(feature_names[feature_x])
                  if feature_x < len(feature_names) else f"Feature {feature_x}")
        ylabel = (clean_feature_name(feature_names[feature_y])
                  if feature_y < len(feature_names) else f"Feature {feature_y}")
    else:
        xlabel, ylabel = f"Feature {feature_x}", f"Feature {feature_y}"
    label_color = "#475569" if interpretability else "#2D3748"
    ax.set_xlabel(xlabel, fontsize=10, color=label_color, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, color=label_color, labelpad=8)
