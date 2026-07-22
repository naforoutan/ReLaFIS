"""Membership-function and parameter-evolution plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from matplotlib.gridspec import GridSpec

from utils.paper_plot_style import (
    AXIS_LABEL_SIZE,
    DOUBLE_COLUMN_WIDTH,
    LEGEND_SIZE,
    TICK_SIZE,
    apply_paper_style,
    make_legend_compact,
    save_paper_figure,
    style_axis,
)
from utils.plot_style import CLASS_FILL, CLASS_SCATTER, RULE_COLORS

# Distinct line styles so rules remain separable in grayscale print.
_RULE_LINESTYLES = ("-", "--", "-.", ":")


def _unwrap_gift_model(model: nn.Module) -> nn.Module:
    """Accept a raw module or an sklearn wrapper exposing ``.model``."""
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "mean") and hasattr(inner, "encode"):
        return inner
    return model


def _softplus_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus matching ``F.softplus``."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -40.0, 20.0))))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _extract_gift_mf_params(model: nn.Module) -> dict:
    """Pull GIFT antecedent parameters onto NumPy."""
    mean = model.mean.detach().cpu().numpy()
    sigma = _softplus_np(model.std.detach().cpu().numpy())
    sigma = np.maximum(sigma, 1e-3)
    literal = _sigmoid_np(model.literal.detach().cpu().numpy())
    slope = model.sigmoid_slope.detach().cpu().numpy()
    temp = _sigmoid_np(model.temp.detach().cpu().numpy())
    weight = _sigmoid_np(model.comb_weight.detach().cpu().numpy())
    return {
        "mean": mean,
        "sigma": sigma,
        "literal": literal,
        "slope": slope,
        "temp": temp,
        "weight": weight,
        "in_features": int(model.in_features),
        "rules": int(model.rules_count),
    }


def _mu_final_1d(
    x: np.ndarray,
    *,
    mu: float,
    sigma: float,
    literal: float,
    slope: float,
    temp: float,
    weight: float,
) -> np.ndarray:
    """Final effective membership for one (feature, rule), matching ``encode()``."""
    x = np.asarray(x, dtype=float)
    # Gaussian branch with equal / not-equal gate
    mu_pos = np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    mu_pos_neg = mu_pos * literal + (1.0 - mu_pos) * (1.0 - literal)
    # Sigmoidal greater / less branch
    mu_greater = _sigmoid_np((x - mu) * slope)
    mu_great_less = mu_greater * temp + (1.0 - mu_greater) * (1.0 - temp)
    return weight * mu_pos_neg + (1.0 - weight) * mu_great_less


def _feature_impact_scores(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Rank features by consequent slope magnitude (fallback: data variance)."""
    if hasattr(model, "local_slopes"):
        slopes = model.local_slopes.detach().cpu().numpy()
        # (rules, in_features, out_features) → per-feature impact
        return np.abs(slopes).sum(axis=(0, 2))
    return np.var(np.asarray(X, dtype=float), axis=0)


def _to_display_coords(
    values_1d: np.ndarray,
    feat_idx: int,
    n_features: int,
    scaler,
) -> np.ndarray:
    """Map model-space coordinates to original units when a scaler is given."""
    values_1d = np.asarray(values_1d, dtype=float).ravel()
    if scaler is None:
        return values_1d
    dummy = np.zeros((values_1d.size, n_features), dtype=float)
    # StandardScaler (and similar affine scalers) invert per-column independently.
    dummy[:, feat_idx] = values_1d
    return np.asarray(scaler.inverse_transform(dummy)[:, feat_idx], dtype=float)


def plot_gift_mfs(
    model,
    X,
    y=None,
    feature_names=None,
    dataset_name="dataset",
    output_path=None,
    top_k=4,
    class_names=None,
    scaler=None,
    bins=25,
    figsize=None,
    dpi=300,
    num_points=500,
    save_path=None,
):
    """Publication grid of class histograms + final effective GIFT MFs.

    Layout mirrors LitANFIS Fig. 9: for each selected feature, a top panel
    shows the class-wise feature distribution and a bottom panel shows the
    final effective membership curves ``mu_final`` for every rule (Gaussian
    branch × relational branch × ``comb_weight``, matching ``encode()``).

    Parameters
    ----------
    model :
        ``GIFT`` module or its sklearn wrapper.
    X :
        Feature matrix in **model input space**.
    y :
        Optional class labels for the histogram panel.
    feature_names :
        Optional feature labels (length = ``in_features``).
    dataset_name :
        Used in the optional export log line.
    output_path / save_path :
        Optional path stem for PDF/PNG(/SVG) export (``save_path`` kept for
        backward compatibility).
    top_k :
        Number of highest-impact features to show (default 4).
    class_names :
        Optional legend labels for classes.
    scaler :
        Optional fitted scaler; when set, axis values are shown in original
        units via ``inverse_transform``.
    bins, figsize, dpi, num_points :
        Histogram / figure / curve resolution controls.
    """
    apply_paper_style()
    core = _unwrap_gift_model(model)
    core.eval()

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")

    params = _extract_gift_mf_params(core)
    in_features = params["in_features"]
    n_rules = params["rules"]
    if X.shape[1] != in_features:
        raise ValueError(
            f"X has {X.shape[1]} features but model expects {in_features}"
        )

    if feature_names is None:
        feature_names = [f"Feature {i + 1}" for i in range(in_features)]
    elif len(feature_names) < in_features:
        feature_names = list(feature_names) + [
            f"Feature {i + 1}" for i in range(len(feature_names), in_features)
        ]

    impact = _feature_impact_scores(core, X)
    order = np.argsort(-impact)
    n_sel = int(min(max(top_k, 1), in_features))
    selected = order[:n_sel].tolist()

    y_arr = None if y is None else np.asarray(y).ravel()
    if y_arr is not None and len(y_arr) != len(X):
        raise ValueError("y must have the same number of rows as X")

    classes = np.unique(y_arr) if y_arr is not None else np.array([])
    if class_names is None:
        class_names = [f"Class {int(c)}" for c in classes]
    else:
        class_names = list(class_names)

    n_cols = 1 if n_sel == 1 else 2
    n_rows = int(np.ceil(n_sel / n_cols))
    if figsize is None:
        figsize = (
            DOUBLE_COLUMN_WIDTH * (0.55 if n_cols == 1 else 1.0),
            max(2.6, 2.75 * n_rows),
        )

    fig = plt.figure(figsize=figsize, facecolor="white", dpi=dpi)
    outer = GridSpec(
        n_rows, n_cols, figure=fig, hspace=0.42, wspace=0.30,
        left=0.08, right=0.98, top=0.96, bottom=0.08,
    )

    out_path = output_path if output_path is not None else save_path

    for panel_i, feat_idx in enumerate(selected):
        row, col = divmod(panel_i, n_cols)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.06)
        ax_hist = fig.add_subplot(inner[0])
        ax_mf = fig.add_subplot(inner[1], sharex=ax_hist)

        x_feat = X[:, feat_idx]
        span_model = float(np.max(x_feat) - np.min(x_feat)) or 1.0
        pad_model = 0.04 * span_model
        x_model = np.linspace(
            float(np.min(x_feat)) - pad_model,
            float(np.max(x_feat)) + pad_model,
            int(num_points),
        )
        x_disp = _to_display_coords(x_feat, feat_idx, in_features, scaler)
        x_plot = _to_display_coords(x_model, feat_idx, in_features, scaler)
        x_lo, x_hi = float(np.min(x_plot)), float(np.max(x_plot))

        # --- top: class-wise histogram (LitANFIS COUNT panel) ---
        style_axis(ax_hist, grid=True)
        bin_edges = np.linspace(x_lo, x_hi, int(bins) + 1)
        if y_arr is None or classes.size == 0:
            ax_hist.hist(
                x_disp, bins=bin_edges, color=CLASS_FILL[0],
                edgecolor=CLASS_SCATTER[0], linewidth=0.6, alpha=0.85,
            )
        else:
            for ci, cls in enumerate(classes):
                mask = y_arr == cls
                label = class_names[ci] if ci < len(class_names) else f"Class {cls}"
                ax_hist.hist(
                    x_disp[mask],
                    bins=bin_edges,
                    color=CLASS_FILL[ci % len(CLASS_FILL)],
                    edgecolor=CLASS_SCATTER[ci % len(CLASS_SCATTER)],
                    linewidth=0.55,
                    alpha=0.72,
                    label=label,
                    histtype="stepfilled",
                )
            leg_c = ax_hist.legend(
                loc="upper right", fontsize=LEGEND_SIZE, framealpha=0.92,
                borderpad=0.25, labelspacing=0.2, handlelength=1.1,
            )
            make_legend_compact(leg_c)

        ax_hist.set_ylabel("Count", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
        ax_hist.tick_params(labelbottom=False, labelsize=TICK_SIZE)
        ax_hist.set_xlim(x_lo, x_hi)

        # --- bottom: final effective membership curves ---
        style_axis(ax_mf, grid=True)
        for r in range(n_rules):
            mu_curve = _mu_final_1d(
                x_model,
                mu=float(params["mean"][feat_idx, r]),
                sigma=float(params["sigma"][feat_idx, r]),
                literal=float(params["literal"][feat_idx, r]),
                slope=float(params["slope"][feat_idx, r]),
                temp=float(params["temp"][feat_idx, r]),
                weight=float(params["weight"][feat_idx, r]),
            )
            color = RULE_COLORS[r % len(RULE_COLORS)]
            ls = _RULE_LINESTYLES[r % len(_RULE_LINESTYLES)]
            ax_mf.plot(
                x_plot, mu_curve, color=color, linestyle=ls, linewidth=1.9,
                label=f"Rule {r + 1}", zorder=3,
            )

        ax_mf.set_ylim(-0.02, 1.05)
        ax_mf.set_ylabel("Membership", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
        ax_mf.set_xlabel(feature_names[feat_idx], fontsize=AXIS_LABEL_SIZE, fontweight="bold")
        ax_mf.tick_params(labelsize=TICK_SIZE)
        ax_mf.set_xlim(x_lo, x_hi)
        leg_r = ax_mf.legend(
            loc="best", fontsize=LEGEND_SIZE, framealpha=0.92,
            borderpad=0.25, labelspacing=0.2, handlelength=1.6,
        )
        make_legend_compact(leg_r)

    # No internal figure title — caption belongs in the manuscript.
    if out_path:
        paths = save_paper_figure(fig, out_path)
        base = Path(out_path)
        if base.suffix:
            base = base.with_suffix("")
        svg_path = base.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        paths = list(paths) + [str(svg_path)]
        print(f"Saved ReLaFIS MF figure ({dataset_name}): {', '.join(paths)}")
    else:
        plt.show()
    return fig


def plot_gift_param_progress(history, model_type):
    """
    Plot mean values of literal, temp, weight, and relax all in one plot.

    Parameters:
    - history: list of dictionaries from model.get_interpretable_params()
    """
    if not history:
        print("No history data available.")
        return

    epochs = [entry["epoch"] for entry in history]

    if model_type == "gift":
        literal_means = [entry["literal_mean"] for entry in history]
        temp_means = [entry["temp_mean"] for entry in history]
        weight_means = [entry["weight_mean"] for entry in history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, literal_means, "b-", label="Literal", linewidth=2)
        plt.plot(epochs, temp_means, "g-", label="Temp", linewidth=2)
        plt.plot(epochs, weight_means, "r-", label="Weight", linewidth=2)

    elif model_type == "litanfis":
        literal_means = [entry["literal_mean"] for entry in history]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, literal_means, "b-", label="Literal", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Mean parameter value")
    plt.title("Evolution of GIFT interpretable parameters")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_litanfis_mfs(model, feature_names=None, num_points=1000, xlim=(-7, 7)):
    """
    Plot membership functions for a LitAnfis model.
    Grid layout: rows = rules, columns = features.

    Parameters:
    - model: trained LitAnfis instance
    - feature_names: list of feature names (optional)
    - num_points: number of points for the smooth curve
    - xlim: tuple (xmin, xmax), default (-7, 7)
    """
    model.eval()

    mean = model.mean.detach().cpu().numpy()
    std = model.std.detach().cpu().numpy()
    literal_raw = model.literal.detach().cpu().numpy()
    literal = 1.0 / (1.0 + np.exp(-literal_raw))

    in_features = model.in_features
    rules = model.rules_count

    if feature_names is None:
        feature_names = [f"F{i}" for i in range(in_features)]

    x_vals = np.linspace(xlim[0], xlim[1], num_points)

    fig, axes = plt.subplots(
        rules, in_features, figsize=(3 * in_features, 3 * rules), sharex=True, sharey=True,
    )
    if rules == 1 and in_features == 1:
        axes = np.array([[axes]])
    elif rules == 1:
        axes = axes.reshape(1, -1)
    elif in_features == 1:
        axes = axes.reshape(-1, 1)

    for rule_idx in range(rules):
        for feat_idx in range(in_features):
            ax = axes[rule_idx, feat_idx]

            mu = mean[feat_idx, rule_idx]
            sigma = std[feat_idx, rule_idx]
            lit = literal[feat_idx, rule_idx]

            g = np.exp(-((x_vals - mu) ** 2) / (2 * sigma ** 2))
            mu_final = lit * g + (1 - lit) * (1 - g)

            ax.plot(x_vals, mu_final, color="b")
            ax.set_ylim(0, 1)
            ax.set_xlim(xlim)
            ax.grid(True, alpha=0.3)

            if rule_idx == 0:
                ax.set_title(feature_names[feat_idx])
            if rule_idx == rules - 1:
                ax.set_xlabel("Input")
            if feat_idx == 0:
                ax.set_ylabel(f"Rule {rule_idx + 1}")

    fig.suptitle("LitANFIS Membership Functions (rows=rules, cols=features)", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_param_evolution_grid(history, param_name, model_type, feature_names=None):
    """
    Plot the evolution of a parameter (literal, temp, weight, relax) over epochs
    in a grid: rows = rules, columns = features.

    Parameters:
    - history: list of dicts from get_interpretable_params()
    - param_name: one of 'literal', 'temp', 'weight', 'relax'
    - model_type: 'gift' or 'litanfis'
    - feature_names: optional list of feature names (length = in_features)
    """
    if not history:
        print("No history data available.")
        return

    first_entry = history[0]
    matrix_key = f"{param_name}_matrix"
    if matrix_key not in first_entry:
        print(
            f"Parameter '{param_name}' not found in history "
            f"(available: {list(first_entry.keys())})"
        )
        return

    matrix = first_entry[matrix_key]
    in_features, rules = matrix.shape

    epochs = [entry["epoch"] for entry in history]
    time_series = [[[] for _ in range(rules)] for _ in range(in_features)]

    for entry in history:
        mat = entry[matrix_key]
        for f in range(in_features):
            for r in range(rules):
                time_series[f][r].append(mat[f, r])

    if feature_names is None:
        feature_names = [f"Feature {f}" for f in range(in_features)]

    fig, axes = plt.subplots(
        rules, in_features, figsize=(3 * in_features, 3 * rules),
        sharex=True, sharey=True,
    )
    if rules == 1 and in_features == 1:
        axes = np.array([[axes]])
    elif rules == 1:
        axes = axes.reshape(1, -1)
    elif in_features == 1:
        axes = axes.reshape(-1, 1)

    for r in range(rules):
        for f in range(in_features):
            ax = axes[r, f]
            values = time_series[f][r]
            ax.plot(epochs, values, "b-", linewidth=1.5)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            if r == 0:
                ax.set_title(feature_names[f], fontsize=10)
            if r == rules - 1:
                ax.set_xlabel("Epoch")
            if f == 0:
                ax.set_ylabel(f"Rule {r + 1}", fontsize=10)

    fig.suptitle(
        f"{param_name.capitalize()} parameter evolution (rows=rules, cols=features)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()
