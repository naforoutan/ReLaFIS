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

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    ListedColormap,
    LinearSegmentedColormap,
    Normalize,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.paper_plot_style import (
    AXIS_LABEL_SIZE,
    DOUBLE_COLUMN_WIDTH,
    LEGEND_SIZE,
    TITLE_SIZE,
    apply_paper_style,
    make_legend_compact,
    panel_label,
    save_figure_artifacts,
    save_paper_figure,
    style_axis,
)
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

# Paper Haberman two-panel figure — Midnight & Coral reference styling.
# Numerical grids/preds unchanged; only display remapping of rule IDs by
# vertical region centroid for visual top/middle/bottom order.
RULE_FILL_COLORS = [
    "#D1E1F0",  # Rule 1 (display) — light powder blue
    "#FDECDA",  # Rule 2 (display) — soft warm cream
    "#919FB9",  # Rule 3 (display) — muted slate blue
]
CLASS_FILL_COLORS = [
    "#D2E2F1",  # predicted class 0
    "#FEEBDA",  # predicted class 1
]
BOUNDARY_COLOR = "#355A73"
POINT_COLORS = {
    0: "#0D1B2A",
    1: "#E76F51",
}

_HABERMAN_TITLE_SIZE = 10.5
_HABERMAN_AXIS_SIZE = 9.5
_HABERMAN_TICK_SIZE = 8.5
_HABERMAN_LEGEND_TITLE_SIZE = 8.5
_HABERMAN_LEGEND_SIZE = 8.0
_HABERMAN_BOUNDARY_LW = 0.80
_HABERMAN_BOUNDARY_ALPHA = 0.95

# Legacy aliases used by standalone panel helpers.
_INTERP_REGION_ALPHA = 1.0
_INTERP_RULE_REGION_FILL = list(RULE_FILL_COLORS)
_INTERP_CLASS_REGION_FILL = list(CLASS_FILL_COLORS)
_INTERP_RULE_BOUNDARY_COLOR = BOUNDARY_COLOR
_INTERP_RULE_BOUNDARY_LW = _HABERMAN_BOUNDARY_LW
_INTERP_RULE_BOUNDARY_ALPHA = _HABERMAN_BOUNDARY_ALPHA
_INTERP_RULE_BOUNDARY_EDGE_COLOR = BOUNDARY_COLOR
_INTERP_RULE_BOUNDARY_EDGE_LW = _HABERMAN_BOUNDARY_LW
_INTERP_RULE_BOUNDARY_EDGE_ALPHA = _HABERMAN_BOUNDARY_ALPHA
_INTERP_DECISION_BOUNDARY_COLOR = BOUNDARY_COLOR
_INTERP_DECISION_BOUNDARY_LW = _HABERMAN_BOUNDARY_LW
_INTERP_DECISION_BOUNDARY_ALPHA = _HABERMAN_BOUNDARY_ALPHA
_INTERP_GRID_RESOLUTION = 500
_INTERP_SCATTER_ALPHA = 0.98
_INTERP_SCATTER_SIZE = 32
_INTERP_SCATTER_EDGE_LW = 0.50
_INTERP_AXIS_MARGIN = 0.12
_INTERP_FIGSIZE = (7.4, 2.75)

_INTERP_CLASS_SCATTER = [POINT_COLORS[0], POINT_COLORS[1]]

_DEFAULT_HABERMAN_CLASS_NAMES = [
    r"Survived $\geq$ 5 years",
    "Died within 5 years",
]


def _haberman_style_axis(ax: plt.Axes) -> None:
    """Compact Midnight & Coral axes (no grid, thin spines)."""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4B5563")
        spine.set_linewidth(0.75)
    ax.tick_params(
        colors="#374151",
        labelsize=_HABERMAN_TICK_SIZE,
        width=0.75,
        length=3.0,
        direction="out",
        pad=2,
    )
    ax.grid(False)


def _visual_rule_order_by_centroid(
    xx: np.ndarray,
    yy: np.ndarray,
    rules: np.ndarray,
    n_rules: int,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """Map original rule IDs → display IDs by mean grid *y* (high→Rule 1).

    Purely visual: does not alter model outputs. Display ID 0 is the
    uppermost region, then 1, then 2, ...
    """
    centroids: List[Tuple[float, int]] = []
    for orig in range(n_rules):
        mask = rules == orig
        mean_y = float(yy[mask].mean()) if np.any(mask) else float("-inf")
        centroids.append((mean_y, orig))
    centroids.sort(key=lambda t: t[0], reverse=True)

    original_to_display: Dict[int, int] = {}
    display_to_original: Dict[int, int] = {}
    for display_id, (_mean_y, orig_id) in enumerate(centroids):
        original_to_display[int(orig_id)] = int(display_id)
        display_to_original[int(display_id)] = int(orig_id)

    displayed = np.empty_like(rules)
    for orig_id, disp_id in original_to_display.items():
        displayed[rules == orig_id] = disp_id
    return displayed, original_to_display, display_to_original


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


def _predict_grid(
    model, grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(predicted_class, confidence, class1_probability)``.

    For binary models ``class1_probability`` is the sigmoid score used for the
    0.5 decision threshold. Predictions are unchanged from prior behaviour.
    """
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
            return (
                pred.cpu().numpy(),
                conf.cpu().numpy(),
                proba.cpu().numpy(),
            )
        proba = torch.softmax(logits, dim=1)
        conf, pred = proba.max(dim=1)
        # Store positive-class score when available; else max probability.
        class1 = proba[:, 1] if proba.shape[1] > 1 else conf
        return (
            pred.cpu().numpy(),
            conf.cpu().numpy(),
            class1.cpu().numpy(),
        )


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
    axis_scale: Optional[str] = None,
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
        axis_scale: ``\"standardized\"`` or ``\"original\"`` for axis wording.
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
    _label_axes(
        ax, feature_x, feature_y, feature_names,
        interpretability=interpretability, axis_scale=axis_scale,
    )
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
    axis_scale: Optional[str] = None,
):
    """Colour the 2D input plane by the model's predicted class.

    See :func:`plot_rule_regions_2d` for the shared arguments.
    """
    xx, yy, grid = _build_projected_grid(
        X, feature_x, feature_y, fixed_values, resolution
    )
    pred, _, _ = _predict_grid(model, grid)
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
    _label_axes(
        ax, feature_x, feature_y, feature_names,
        interpretability=interpretability, axis_scale=axis_scale,
    )
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
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
    axis_scale: str = "standardized",
    *,
    metadata: Optional[Dict[str, Any]] = None,
    scaler=None,
    optimizer=None,
    preprocessing_bundle: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    run_index: Optional[int] = None,
    fold_index: Optional[int] = None,
    train_indices: Optional[Sequence[int]] = None,
    val_indices: Optional[Sequence[int]] = None,
    test_indices: Optional[Sequence[int]] = None,
    source_split: Optional[str] = None,
    overwrite_artifacts: bool = False,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
):
    """Side-by-side rule regions and decision boundary on the same projection.

    Numerical grids and predictions are computed once with the existing helpers
    (``_build_projected_grid``, ``_dominant_rule``, ``_predict_grid``). Only
    layout / typography / export paths are publication-oriented.

    When ``save_path`` is set, a full reproducibility bundle is written under
    ``artifacts/figures/HabermanBoundary/<run_id>/`` (or the parent of
    ``save_path``).
    """
    del suptitle  # caption belongs in LaTeX
    if axis_scale not in {"standardized", "original"}:
        raise ValueError(
            f"axis_scale must be 'standardized' or 'original', got {axis_scale!r}"
        )
    if class_names is None:
        class_names = list(_DEFAULT_HABERMAN_CLASS_NAMES)
    if figsize is None:
        figsize = _INTERP_FIGSIZE

    meta = dict(metadata or {})
    if seed is not None:
        meta.setdefault("seed", seed)
    if run_index is not None:
        meta.setdefault("run_index", run_index)
    if fold_index is not None:
        meta.setdefault("fold_index", fold_index)
    if epoch is not None:
        meta.setdefault("epoch", epoch)
    if metrics is not None:
        meta.setdefault("metrics", metrics)

    grid_resolution = max(resolution, _INTERP_GRID_RESOLUTION)
    xx, yy, grid = _build_projected_grid(
        X, feature_x, feature_y, fixed_values, grid_resolution
    )
    rules = _dominant_rule(model, grid).reshape(xx.shape)
    pred, conf, proba_grid = _predict_grid(model, grid)
    pred = pred.reshape(xx.shape)
    conf = conf.reshape(xx.shape)
    proba_grid = proba_grid.reshape(xx.shape)

    X_arr = np.asarray(X, dtype=np.float64)
    sample_rules = _dominant_rule(model, X_arr)
    sample_pred, sample_conf, sample_proba = _predict_grid(model, X_arr)
    n_rules_model = int(_unwrap_model(model).rules_count)

    # Draw first (no disk write); artifact helper owns PDF/PNG + bundle.
    fig, axes = draw_interpretability_panels(
        xx, yy, rules, pred, X_arr, y,
        feature_x=feature_x,
        feature_y=feature_y,
        feature_names=feature_names,
        class_names=class_names,
        figsize=figsize,
        axis_scale=axis_scale,
        save_path=None,
        conf=proba_grid,
        sample_pred=sample_pred,
        sample_rules=sample_rules,
        n_rules=n_rules_model,
        metadata=meta,
        show=False,
    )

    if save_path:
        from utils.figure_artifacts import (
            extract_model_config,
            resolve_haberman_run_dir,
            save_haberman_figure_artifacts,
        )

        def _haberman_collection_root(path: Path) -> Path:
            """Return ``.../HabermanBoundary`` collection directory."""
            parts = list(path.resolve().parts)
            if "HabermanBoundary" in parts:
                idx = parts.index("HabermanBoundary")
                return Path(*parts[: idx + 1])
            # treat *path* as a file stem under the collection root
            return path.parent / "HabermanBoundary"

        run_parent = _haberman_collection_root(Path(save_path))
        run_dir = resolve_haberman_run_dir(
            run_parent,
            seed=meta.get("seed"),
            run_index=meta.get("run_index", 0 if meta.get("seed") is not None else None),
            fold_index=meta.get("fold_index"),
            overwrite=overwrite_artifacts,
        )

        # Recompute display remapping (same helper used in draw).
        displayed_rules, o2d, d2o = _visual_rule_order_by_centroid(
            xx, yy, rules, n_rules_model,
        )
        xlabel = axes[0].get_xlabel()
        ylabel = axes[0].get_ylabel()
        xlim = axes[0].get_xlim()
        ylim = axes[0].get_ylim()

        xs = X_arr[:, feature_x]
        ys = X_arr[:, feature_y]
        y_arr = np.asarray(y).ravel() if y is not None else np.full(len(xs), np.nan)
        sample_display = np.array(
            [o2d.get(int(r), int(r)) for r in sample_rules], dtype=np.int64,
        )

        points_df = pd.DataFrame({
            "sample_index": np.arange(len(xs)),
            "x_display": xs,
            "y_display": ys,
            "true_label": y_arr,
            "predicted_label": sample_pred,
            "predicted_probability": sample_proba,
            "dominant_rule_original": sample_rules,
            "dominant_rule_display": sample_display,
            "source_split": source_split if source_split is not None else "train",
        })

        grids = {
            "xx": np.asarray(xx),
            "yy": np.asarray(yy),
            "dominant_rule_grid_original": np.asarray(rules),
            "dominant_rule_grid_display": np.asarray(displayed_rules),
            "dominant_rule_grid": np.asarray(rules),  # alias
            "predicted_class_grid": np.asarray(pred),
            "predicted_probability_grid": np.asarray(proba_grid),
            "prediction_confidence_grid": np.asarray(conf),
            "decision_boundary_level": np.asarray([0.5], dtype=np.float64),
            "x_display": xs,
            "y_display": ys,
            "true_labels": y_arr,
            "predicted_labels": np.asarray(sample_pred),
            "predicted_probabilities": np.asarray(sample_proba),
            "dominant_rule_per_sample_original": np.asarray(sample_rules),
            "dominant_rule_per_sample_display": sample_display,
        }

        mcfg = dict(model_config or extract_model_config(model))
        plot_config = {
            "figure_name": "HabermanBoundary",
            "figure_stem": "HabermanBoundary",
            "figure_size": list(figsize),
            "point_colors": {str(k): v for k, v in POINT_COLORS.items()},
            "class_colors": [POINT_COLORS[0], POINT_COLORS[1]],
            "rule_colors": list(RULE_FILL_COLORS[:n_rules_model]),
            "class_fill_colors": list(CLASS_FILL_COLORS),
            "boundary_color": BOUNDARY_COLOR,
            "marker": "o",
            "marker_size": _INTERP_SCATTER_SIZE,
            "marker_alpha": _INTERP_SCATTER_ALPHA,
            "marker_edgecolor": "white",
            "marker_edgewidth": _INTERP_SCATTER_EDGE_LW,
            "boundary_width": _HABERMAN_BOUNDARY_LW,
            "rule_boundary_width": _HABERMAN_BOUNDARY_LW,
            "contour_levels_decision": [0.5],
            "panel_titles": [
                "(a) Dominant rule regions",
                "(b) Predicted class regions",
            ],
            "axis_labels": {"x": xlabel, "y": ylabel},
            "x_limits": list(xlim),
            "y_limits": list(ylim),
            "font_family": ["Times New Roman", "Times", "DejaVu Serif"],
            "font_sizes": {
                "panel_title": _HABERMAN_TITLE_SIZE,
                "axis_label": _HABERMAN_AXIS_SIZE,
                "tick": _HABERMAN_TICK_SIZE,
                "legend_title": _HABERMAN_LEGEND_TITLE_SIZE,
                "legend": _HABERMAN_LEGEND_SIZE,
            },
            "legend_locations": {"rules": "upper right", "classes": "upper right"},
            "legend_labels": {
                "rules": [f"Rule {i + 1}" for i in range(n_rules_model)],
                "classes": list(class_names),
            },
            "class_mapping": {
                "0": class_names[0] if len(class_names) > 0 else "class_0",
                "1": class_names[1] if len(class_names) > 1 else "class_1",
            },
            "original_to_display_rule_mapping": {str(k): v for k, v in o2d.items()},
            "display_to_original_rule_mapping": {str(k): v for k, v in d2o.items()},
            "axis_scale": axis_scale,
            "feature_x": int(feature_x),
            "feature_y": int(feature_y),
            "feature_names": list(feature_names) if feature_names is not None else None,
            "n_rules": n_rules_model,
            "mesh_resolution": int(grid_resolution),
            "decision_threshold": 0.5,
            "output_filenames": {
                "pdf": "HabermanBoundary.pdf",
                "png": "HabermanBoundary.png",
            },
        }

        full_meta = {
            **meta,
            "dataset": meta.get("dataset_name", meta.get("dataset", "Haberman")),
            "dataset_name": meta.get("dataset_name", "Haberman"),
            "model_name": meta.get("model_name", _unwrap_model(model).__class__.__name__),
            "number_of_rules": n_rules_model,
            "feature_names": list(feature_names) if feature_names is not None else None,
            "feature_indices": {"x": int(feature_x), "y": int(feature_y)},
            "selected_x_feature_index": int(feature_x),
            "selected_y_feature_index": int(feature_y),
            "selected_x_feature_name": (
                feature_names[feature_x]
                if feature_names is not None and feature_x < len(feature_names)
                else None
            ),
            "selected_y_feature_name": (
                feature_names[feature_y]
                if feature_names is not None and feature_y < len(feature_names)
                else None
            ),
            "class_mapping": plot_config["class_mapping"],
            "rule_display_mapping": plot_config["original_to_display_rule_mapping"],
            "axis_scale": axis_scale,
            "mesh_resolution": int(grid_resolution),
            "decision_threshold": 0.5,
            "source_script": "utils/decision_boundary.py:plot_interpretability_2d",
            "train_indices": (
                list(map(int, train_indices)) if train_indices is not None else None
            ),
            "val_indices": (
                list(map(int, val_indices)) if val_indices is not None else None
            ),
            "test_indices": (
                list(map(int, test_indices)) if test_indices is not None else None
            ),
        }

        root = Path(__file__).resolve().parents[1]
        repro_src = root / "scripts" / "reproduce_haberman_boundary.py"
        written = save_haberman_figure_artifacts(
            output_dir=run_dir,
            fig=fig,
            points_df=points_df,
            grids=grids,
            plot_config=plot_config,
            metadata=full_meta,
            model=model,
            model_config=mcfg,
            scaler=scaler,
            optimizer=optimizer,
            preprocessing_bundle=preprocessing_bundle,
            reproduce_script_src=repro_src if repro_src.is_file() else None,
        )
        print(
            "Saved HabermanBoundary artifacts: "
            + ", ".join(f"{k}={v}" for k, v in written.items())
        )

    return fig, axes


def draw_interpretability_panels(
    xx: np.ndarray,
    yy: np.ndarray,
    rules: np.ndarray,
    pred: np.ndarray,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    feature_x: int = 0,
    feature_y: int = 1,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = _INTERP_FIGSIZE,
    axis_scale: str = "standardized",
    save_path: Optional[str] = None,
    conf: Optional[np.ndarray] = None,
    sample_pred: Optional[np.ndarray] = None,
    sample_rules: Optional[np.ndarray] = None,
    n_rules: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    show: bool = True,
):
    """Draw the Haberman-style two-panel figure from precomputed arrays."""
    apply_paper_style()
    if class_names is None:
        class_names = list(_DEFAULT_HABERMAN_CLASS_NAMES)

    X = np.asarray(X, dtype=np.float64)
    rules_original = np.asarray(rules)
    pred = np.asarray(pred)
    n_rules = int(n_rules) if n_rules is not None else (
        int(np.max(rules_original)) + 1 if rules_original.size else 1
    )
    n_rules = max(n_rules, int(np.max(rules_original)) + 1 if rules_original.size else 1)
    n_classes = int(np.max(pred)) + 1 if pred.size else 2
    n_classes = max(n_classes, 2)

    displayed_rules, original_to_display, display_to_original = (
        _visual_rule_order_by_centroid(xx, yy, rules_original, n_rules)
    )

    xlabel = _axis_label_text(feature_x, feature_names, axis_scale=axis_scale)
    ylabel = _axis_label_text(feature_y, feature_names, axis_scale=axis_scale)

    fig, axes = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True, facecolor="white",
    )

    rule_fills = list(RULE_FILL_COLORS)
    while len(rule_fills) < n_rules:
        rule_fills.append(rule_fills[len(rule_fills) % len(RULE_FILL_COLORS)])
    rule_fills = rule_fills[:n_rules]
    rule_levels = np.arange(-0.5, len(rule_fills) + 0.5, 1)
    rule_cmap = ListedColormap(rule_fills)
    rule_norm = BoundaryNorm(rule_levels, rule_cmap.N)

    class_fills = list(CLASS_FILL_COLORS)
    while len(class_fills) < n_classes:
        class_fills.append(class_fills[len(class_fills) % len(CLASS_FILL_COLORS)])
    class_fills = class_fills[:n_classes]
    class_levels = np.arange(-0.5, len(class_fills) + 0.5, 1)
    class_cmap = ListedColormap(class_fills)
    class_norm = BoundaryNorm(class_levels, class_cmap.N)

    # --- Panel (a): dominant rule regions (display-ordered) ---------------
    ax0 = axes[0]
    _haberman_style_axis(ax0)
    ax0.contourf(
        xx, yy, displayed_rules,
        levels=rule_levels,
        cmap=rule_cmap,
        norm=rule_norm,
        alpha=1.0,
        antialiased=True,
        corner_mask=True,
        zorder=0,
    )
    if n_rules > 1:
        boundary_levels = np.arange(0.5, n_rules - 0.5 + 1, 1)
        ax0.contour(
            xx, yy, displayed_rules, levels=boundary_levels,
            colors=BOUNDARY_COLOR,
            linewidths=_HABERMAN_BOUNDARY_LW,
            alpha=_HABERMAN_BOUNDARY_ALPHA,
            antialiased=True,
            zorder=2,
        )
    _scatter_true_classes(ax0, X, y, feature_x, feature_y, class_names, with_legend=False)
    _set_interp_axis_limits(ax0, X, feature_x, feature_y)
    ax0.set_xlabel(
        xlabel, fontsize=_HABERMAN_AXIS_SIZE, fontweight="bold", labelpad=3,
    )
    ax0.set_ylabel(
        ylabel, fontsize=_HABERMAN_AXIS_SIZE, fontweight="bold", labelpad=3,
    )
    ax0.set_title(
        "(a) Dominant rule regions",
        fontsize=_HABERMAN_TITLE_SIZE, fontweight="bold", pad=4,
    )

    rule_handles = [
        Patch(facecolor=color, edgecolor="none", label=f"Rule {index + 1}")
        for index, color in enumerate(rule_fills)
    ]
    legend_left = ax0.legend(
        handles=rule_handles,
        title="Dominant rule",
        loc="upper right",
        fontsize=_HABERMAN_LEGEND_SIZE,
        title_fontsize=_HABERMAN_LEGEND_TITLE_SIZE,
        frameon=True,
        framealpha=0.90,
        borderpad=0.25,
        labelspacing=0.20,
        handlelength=1.0,
        handletextpad=0.30,
        borderaxespad=0.35,
    )
    _haberman_compact_legend(legend_left)

    # --- Panel (b): predicted class regions -------------------------------
    ax1 = axes[1]
    _haberman_style_axis(ax1)
    ax1.contourf(
        xx, yy, pred,
        levels=class_levels,
        cmap=class_cmap,
        norm=class_norm,
        alpha=1.0,
        antialiased=True,
        corner_mask=True,
        zorder=0,
    )
    boundary_levels = np.arange(0.5, n_classes - 0.5 + 1, 1)
    if len(boundary_levels) > 0:
        ax1.contour(
            xx, yy, pred, levels=boundary_levels,
            colors=BOUNDARY_COLOR,
            linewidths=_HABERMAN_BOUNDARY_LW,
            linestyles="-",
            alpha=_HABERMAN_BOUNDARY_ALPHA,
            antialiased=True,
            zorder=2,
        )
    _scatter_true_classes(ax1, X, y, feature_x, feature_y, class_names, with_legend=True)
    _set_interp_axis_limits(ax1, X, feature_x, feature_y)
    ax1.set_xlabel(
        xlabel, fontsize=_HABERMAN_AXIS_SIZE, fontweight="bold", labelpad=3,
    )
    ax1.set_ylabel(
        ylabel, fontsize=_HABERMAN_AXIS_SIZE, fontweight="bold", labelpad=3,
    )
    ax1.set_title(
        "(b) Predicted class regions",
        fontsize=_HABERMAN_TITLE_SIZE, fontweight="bold", pad=4,
    )

    # Shared axis limits across panels.
    xlim, ylim = ax0.get_xlim(), ax0.get_ylim()
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    if save_path:
        paths = save_paper_figure(fig, save_path)
        _export_interpretability_artifacts(
            save_path=save_path,
            xx=xx, yy=yy,
            rules=rules_original,
            displayed_rules=displayed_rules,
            pred=pred, conf=conf,
            X=X, y=y,
            feature_x=feature_x, feature_y=feature_y,
            feature_names=feature_names,
            class_names=class_names,
            axis_scale=axis_scale,
            figsize=figsize,
            xlim=xlim, ylim=ylim,
            xlabel=xlabel, ylabel=ylabel,
            sample_pred=sample_pred,
            sample_rules=sample_rules,
            metadata=metadata,
            n_rules=n_rules,
            rule_colors=rule_fills,
            class_fill_colors=class_fills,
            original_to_display=original_to_display,
            display_to_original=display_to_original,
        )
        print(f"Saved interpretability figure: {', '.join(paths)}")
    elif show:
        plt.show()
    return fig, axes


def _haberman_compact_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#D1D5DB")
    frame.set_alpha(0.90)
    frame.set_linewidth(0.6)
    title = legend.get_title()
    if title is not None:
        title.set_fontsize(_HABERMAN_LEGEND_TITLE_SIZE)
        title.set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontsize(_HABERMAN_LEGEND_SIZE)
        text.set_fontweight("normal")


def redraw_interpretability_from_artifacts(
    figure_dir: Union[str, Path],
    *,
    save_path: Optional[str] = None,
):
    """Regenerate the Haberman boundary figure from saved CSV/NPZ/JSON only."""
    from utils.paper_plot_style import load_figure_artifacts

    figure_dir = Path(figure_dir)
    arts = load_figure_artifacts(figure_dir)
    grids = arts.get("grids")
    cfg = arts.get("plot_config") or {}
    points = arts.get("points_df")
    if grids is None or points is None:
        raise FileNotFoundError(
            f"Missing grids/points artifacts under {figure_dir}"
        )

    X = np.column_stack([
        points["x_display"].to_numpy(dtype=np.float64),
        points["y_display"].to_numpy(dtype=np.float64),
    ])
    # Pad to full feature matrix slots used only for indexing feature_x/y.
    feature_x = int(cfg.get("feature_x", 0))
    feature_y = int(cfg.get("feature_y", 1))
    n_features = max(feature_x, feature_y) + 1
    X_full = np.zeros((len(points), n_features), dtype=np.float64)
    X_full[:, feature_x] = points["x_display"].to_numpy(dtype=np.float64)
    X_full[:, feature_y] = points["y_display"].to_numpy(dtype=np.float64)

    y = points["true_label"].to_numpy() if "true_label" in points.columns else None
    sample_pred = (
        points["predicted_label"].to_numpy()
        if "predicted_label" in points.columns else None
    )
    sample_rules = (
        points["dominant_rule"].to_numpy()
        if "dominant_rule" in points.columns else None
    )

    out = save_path or str(figure_dir / cfg.get("figure_stem", figure_dir.name))
    conf = grids.get("predicted_probability_grid")
    # Prefer original rule IDs; display remapping is reapplied in draw.
    rules = grids.get(
        "dominant_rule_grid_original",
        grids.get("dominant_rule_grid"),
    )
    return draw_interpretability_panels(
        grids["xx"], grids["yy"],
        rules, grids["predicted_class_grid"],
        X_full, y,
        feature_x=feature_x,
        feature_y=feature_y,
        feature_names=cfg.get("feature_names"),
        class_names=cfg.get("class_labels"),
        figsize=_INTERP_FIGSIZE,
        axis_scale=cfg.get("axis_scale", "standardized"),
        save_path=out,
        conf=conf,
        sample_pred=sample_pred,
        sample_rules=sample_rules,
        n_rules=cfg.get("n_rules"),
        metadata=arts.get("metadata"),
    )


def _export_interpretability_artifacts(
    *,
    save_path: str,
    xx, yy, rules, displayed_rules, pred, conf,
    X, y,
    feature_x, feature_y,
    feature_names, class_names,
    axis_scale, figsize, xlim, ylim, xlabel, ylabel,
    sample_pred, sample_rules, metadata,
    n_rules=None,
    rule_colors=None,
    class_fill_colors=None,
    original_to_display=None,
    display_to_original=None,
) -> None:
    base = Path(save_path)
    if base.suffix:
        base = base.with_suffix("")
    figure_dir = base.parent
    stem = base.name

    xs = np.asarray(X)[:, feature_x]
    ys = np.asarray(X)[:, feature_y]
    y_arr = np.asarray(y).ravel() if y is not None else np.full(len(xs), np.nan)
    if sample_pred is None:
        sample_pred = np.full(len(xs), np.nan)
    if sample_rules is None:
        sample_rules = np.full(len(xs), np.nan)

    o2d = original_to_display or {}
    sample_rules_arr = np.asarray(sample_rules).ravel()
    sample_display = np.array([
        o2d.get(int(r), int(r)) if np.isfinite(r) else r for r in sample_rules_arr
    ], dtype=float)

    points_df = pd.DataFrame({
        "sample_index": np.arange(len(xs)),
        "x_display": xs,
        "y_display": ys,
        "true_label": y_arr,
        "predicted_label": np.asarray(sample_pred).ravel(),
        "dominant_rule": sample_rules_arr,
        "displayed_rule": sample_display,
        "point_color": [
            POINT_COLORS.get(int(c), "#000000") if np.isfinite(c) else ""
            for c in y_arr
        ],
    })

    grid_dict = {
        "xx": np.asarray(xx),
        "yy": np.asarray(yy),
        "dominant_rule_grid": np.asarray(rules),  # original model rule IDs
        "dominant_rule_grid_original": np.asarray(rules),
        "displayed_rule_grid": np.asarray(displayed_rules),
        "predicted_class_grid": np.asarray(pred),
    }
    if conf is not None:
        grid_dict["predicted_probability_grid"] = np.asarray(conf)

    n_rules_export = int(n_rules) if n_rules is not None else (
        int(np.max(rules)) + 1 if np.asarray(rules).size else 0
    )
    plot_config = {
        "figure_name": stem,
        "figure_stem": stem,
        "figure_size": list(figsize),
        "x_label": xlabel,
        "y_label": ylabel,
        "x_limits": list(xlim),
        "y_limits": list(ylim),
        "feature_x": int(feature_x),
        "feature_y": int(feature_y),
        "feature_names": list(feature_names) if feature_names is not None else None,
        "class_labels": list(class_names) if class_names is not None else None,
        "n_rules": n_rules_export,
        "rule_labels": [f"Rule {i + 1}" for i in range(n_rules_export)],
        "class_colors": [POINT_COLORS[0], POINT_COLORS[1]],
        "point_colors": {str(k): v for k, v in POINT_COLORS.items()},
        "rule_colors": list(rule_colors or RULE_FILL_COLORS),
        "class_fill_colors": list(class_fill_colors or CLASS_FILL_COLORS),
        "boundary_color": BOUNDARY_COLOR,
        "scatter_size": _INTERP_SCATTER_SIZE,
        "scatter_alpha": _INTERP_SCATTER_ALPHA,
        "scatter_edge_linewidth": _INTERP_SCATTER_EDGE_LW,
        "scatter_edgecolor": "white",
        "background_alpha": 1.0,
        "rule_boundary_linewidth": _HABERMAN_BOUNDARY_LW,
        "decision_boundary_linewidth": _HABERMAN_BOUNDARY_LW,
        "font_sizes": {
            "panel_title": _HABERMAN_TITLE_SIZE,
            "axis_label": _HABERMAN_AXIS_SIZE,
            "tick": _HABERMAN_TICK_SIZE,
            "legend_title": _HABERMAN_LEGEND_TITLE_SIZE,
            "legend": _HABERMAN_LEGEND_SIZE,
        },
        "original_to_display_rule_mapping": {
            str(k): v for k, v in (original_to_display or {}).items()
        },
        "display_to_original_rule_mapping": {
            str(k): v for k, v in (display_to_original or {}).items()
        },
        "legend_location": "upper right",
        "title_strings": [
            "(a) Dominant rule regions",
            "(b) Predicted class regions",
        ],
        "axis_scale": axis_scale,
        "source_script": "utils/decision_boundary.py",
        "dataset_name": (metadata or {}).get("dataset_name", "Haberman"),
        "seed": (metadata or {}).get("seed"),
        "fold_or_run": (metadata or {}).get("fold_or_run"),
        "output_filenames": [f"{stem}.pdf", f"{stem}.png"],
    }

    save_figure_artifacts(
        figure_dir,
        figure_stem=stem,
        points_df=points_df,
        grid_dict=grid_dict,
        plot_config=plot_config,
        metadata=metadata,
    )


def _class_label(cls: int, class_names: Optional[List[str]]) -> str:
    if class_names is not None and 0 <= int(cls) < len(class_names):
        return class_names[int(cls)]
    if 0 <= int(cls) < len(_DEFAULT_HABERMAN_CLASS_NAMES):
        return _DEFAULT_HABERMAN_CLASS_NAMES[int(cls)]
    return f"Class {int(cls)}"


def _scatter_true_classes(
    ax: plt.Axes,
    X: np.ndarray,
    y: Optional[np.ndarray],
    feature_x: int,
    feature_y: int,
    class_names: Optional[List[str]],
    *,
    with_legend: bool,
) -> None:
    X = np.asarray(X)
    xs, ys = X[:, feature_x], X[:, feature_y]
    if y is None:
        ax.scatter(
            xs, ys, c="#4B5563", s=_INTERP_SCATTER_SIZE,
            alpha=_INTERP_SCATTER_ALPHA, edgecolors="white",
            linewidths=_INTERP_SCATTER_EDGE_LW, marker="o", zorder=3,
        )
        return

    y = np.asarray(y).ravel()
    classes = np.unique(y)
    for i, cls in enumerate(classes):
        mask = y == cls
        color = POINT_COLORS.get(int(cls), _INTERP_CLASS_SCATTER[i % len(_INTERP_CLASS_SCATTER)])
        ax.scatter(
            xs[mask], ys[mask], color=color,
            s=_INTERP_SCATTER_SIZE, alpha=_INTERP_SCATTER_ALPHA,
            edgecolors="white", linewidths=_INTERP_SCATTER_EDGE_LW,
            marker="o",
            label=_class_label(int(cls), class_names), zorder=3,
        )
    if with_legend:
        legend = ax.legend(
            title="True class",
            loc="upper right",
            fontsize=_HABERMAN_LEGEND_SIZE,
            title_fontsize=_HABERMAN_LEGEND_TITLE_SIZE,
            frameon=True,
            framealpha=0.90,
            borderpad=0.25,
            labelspacing=0.20,
            handlelength=1.0,
            handletextpad=0.30,
            borderaxespad=0.35,
        )
        _haberman_compact_legend(legend)


def _overlay_scatter(
    ax, X, y, feature_x, feature_y, class_names, scatter, *,
    interpretability: bool = False,
    class_legend_y: float = 1.0,
):
    if not scatter:
        return
    if interpretability:
        _scatter_true_classes(
            ax, X, y, feature_x, feature_y, class_names,
            with_legend=False,
        )
        return
    X = np.asarray(X)
    xs, ys = X[:, feature_x], X[:, feature_y]
    if y is None:
        ax.scatter(
            xs, ys, c="#2D3748", s=28, alpha=0.55,
            edgecolors="white", linewidths=0.65, zorder=4,
        )
        return
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    for i, cls in enumerate(classes):
        mask = y == cls
        label = _class_label(int(cls), class_names)
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
    """Compact top-right legend for dark scatter bubbles (legacy helper)."""
    handles = []
    labels = []
    for cls in classes:
        label = _class_label(int(cls), class_names)
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


def _axis_label_text(
    feature_idx: int,
    feature_names: Optional[List[str]],
    *,
    axis_scale: Optional[str] = None,
) -> str:
    """Human-readable axis label; mark standardized model space when needed."""
    if feature_names is not None and feature_idx < len(feature_names):
        label = clean_feature_name(feature_names[feature_idx])
    else:
        label = f"Feature {feature_idx}"

    low = label.lower()
    if "nod" in low:
        label = "Positive axillary nodes"
    elif "age" in low:
        label = "Age"

    if axis_scale == "original":
        if label == "Age":
            return "Age (years)"
        return label
    if axis_scale == "standardized":
        if label == "Age":
            return "Age (standardized)"
        if "(standardized)" not in label.lower():
            return f"{label} (standardized)"
    return label


def _label_axes(
    ax,
    feature_x,
    feature_y,
    feature_names,
    *,
    interpretability: bool = False,
    axis_scale: Optional[str] = None,
):
    xlabel = _axis_label_text(feature_x, feature_names, axis_scale=axis_scale)
    ylabel = _axis_label_text(feature_y, feature_names, axis_scale=axis_scale)
    if interpretability:
        ax.set_xlabel(
            xlabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=4,
        )
        ax.set_ylabel(
            ylabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=4,
        )
        return
    label_color = "#2D3748"
    ax.set_xlabel(xlabel, fontsize=10, color=label_color, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, color=label_color, labelpad=8)
