"""Shared publication-ready matplotlib / seaborn styling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

# Rose → mauve → deep navy (reference bubble aesthetic).
ROSE_NAVY_SEQUENCE = [
    "#E8B4BC",
    "#D98BA3",
    "#B76E91",
    "#8E4D76",
    "#5A365F",
    "#2E2148",
]

HIGHLIGHT_MODEL_KEY = "gift"


def rose_navy_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("rose_navy", ROSE_NAVY_SEQUENCE)


# Model-comparison accents (non-bubble charts).
ELEGANT = {
    "navy": "#2E2148",
    "deep_blue": "#2E5A88",
    "cyan": "#3A9FBF",
    "teal": "#2A9D8F",
    "emerald": "#3D8B6E",
    "violet": "#6A5ACD",
    "pink": "#B5838D",
    "warm_orange": "#E76F51",
    "accent": "#C1121F",
}
ELEGANT_SEQUENCE = [
    ELEGANT["deep_blue"],
    ELEGANT["teal"],
    ELEGANT["violet"],
    ELEGANT["warm_orange"],
    ELEGANT["emerald"],
    ELEGANT["cyan"],
    ELEGANT["pink"],
    ELEGANT["navy"],
]

MODEL_COLORS: Dict[str, str] = {
    "gift": ELEGANT["accent"],
    "giftshifter": ELEGANT["accent"],  # alias
    "giftshift": ELEGANT["warm_orange"],
    "giftshiftentropy": "#E9C46A",
    "litanfis": ELEGANT["deep_blue"],
    "griffin": ELEGANT["emerald"],
    "unfis": ELEGANT["violet"],
    "anfis": "#8D99AE",
    "gift": ELEGANT["warm_orange"],
}
FALLBACK_PALETTE = ELEGANT_SEQUENCE

RULE_COLORS = ELEGANT_SEQUENCE[:6]
CLASS_FILL = ["#D4E4F2", "#F5D0D0", "#D5ECD8", "#E5DAF2"]
CLASS_SCATTER = [ELEGANT["navy"], ELEGANT["accent"], ELEGANT["emerald"], ELEGANT["violet"]]

PANEL_FACE = "#FAFBFC"
GRID_COLOR = "#E2E8F0"
TICK_COLOR = "#475569"
TITLE_COLOR = "#0F172A"
POSITIVE_BAR = ELEGANT["emerald"]
NEGATIVE_BAR = "#C95D63"
NEUTRAL_LINE = "#64748B"

HEATMAP_CMAP = "mako_r"   # dark = active / low relaxation
DECISION_CMAP = "crest"


def apply_plot_style() -> None:
    """Seaborn whitegrid theme + project typography and spacing."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=0.95,
        palette=ELEGANT_SEQUENCE,
    )
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": PANEL_FACE,
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": TITLE_COLOR,
        "axes.titlecolor": TITLE_COLOR,
        "axes.titleweight": "600",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.labelweight": "medium",
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "-",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.65,
        "xtick.color": TICK_COLOR,
        "ytick.color": TICK_COLOR,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.97,
        "legend.edgecolor": "#E2E8F0",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _normalize_model_key(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def model_color(name: str, index: int = 0) -> str:
    key = _normalize_model_key(name)
    return MODEL_COLORS.get(key, FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)])


def is_highlight_model(name: str) -> bool:
    return HIGHLIGHT_MODEL_KEY in _normalize_model_key(name)


def model_display_name(name: str) -> str:
    return name.upper() if name.islower() else name


def model_line_kwargs(name: str) -> dict:
    highlight = is_highlight_model(name)
    return {
        "linewidth": 2.6 if highlight else 1.7,
        "markersize": 7.5 if highlight else 5.5,
        "zorder": 10 if highlight else 5,
        "alpha": 0.95 if highlight else 0.88,
    }


def rule_cmap(n_rules: int) -> ListedColormap:
    return ListedColormap([RULE_COLORS[i % len(RULE_COLORS)] for i in range(n_rules)])


def class_cmap(n_classes: int) -> ListedColormap:
    n = max(n_classes, 2)
    return ListedColormap([CLASS_FILL[i % len(CLASS_FILL)] for i in range(n)])


def scale_bubble_sizes(
    values: Sequence[float],
    s_min: float = 28.0,
    s_max: float = 320.0,
) -> np.ndarray:
    """Map values to matplotlib ``s`` (points²) with sqrt scaling."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return v
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if np.isclose(vmax, vmin):
        return np.full_like(v, (s_min + s_max) / 2.0)
    norm = (v - vmin) / (vmax - vmin)
    return s_min + np.sqrt(norm) * (s_max - s_min)


def style_axes_minimal(ax: plt.Axes) -> None:
    """White canvas, light spines, no grid — bubble / scatter reference look."""
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D1D5DB")
        spine.set_linewidth(0.65)
    ax.tick_params(colors=TICK_COLOR, labelsize=9, length=3.5, width=0.65)


def style_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.set_facecolor(PANEL_FACE)
    if grid:
        ax.grid(True, linestyle="-", linewidth=0.45, alpha=0.65, color=GRID_COLOR, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=TICK_COLOR, labelsize=9)


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#E2E8F0")
    frame.set_alpha(0.97)
    frame.set_linewidth(0.8)
    title = legend.get_title()
    if title is not None:
        title.set_fontsize(9)
        title.set_fontweight("semibold")
        title.set_color(TITLE_COLOR)


def panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="600",
                 color=TITLE_COLOR, pad=10)


def _compact_bubble_legends(
    ax: plt.Axes,
    size_values: Sequence[float],
    color_values: Sequence[float],
    size_label: str,
    color_label: str,
    cmap: LinearSegmentedColormap,
) -> None:
    """Top-right size + colour reference legends (representative ticks only)."""
    v_size = np.asarray(size_values, dtype=float)
    v_size = v_size[np.isfinite(v_size)]
    v_color = np.asarray(color_values, dtype=float)
    v_color = v_color[np.isfinite(v_color)]
    if len(v_size) == 0 or len(v_color) == 0:
        return

    size_ticks = np.linspace(np.nanmin(v_size), np.nanmax(v_size), 3)
    color_ticks = np.linspace(np.nanmin(v_color), np.nanmax(v_color), 3)
    norm = Normalize(vmin=np.nanmin(v_color), vmax=np.nanmax(v_color))

    size_handles = [
        Line2D(
            [], [], linestyle="none", marker="o",
            markersize=np.sqrt(s / np.pi) * 0.48,
            markerfacecolor=ROSE_NAVY_SEQUENCE[3],
            markeredgecolor="white", markeredgewidth=0.65, alpha=0.75,
        )
        for s in scale_bubble_sizes(size_ticks, s_min=28, s_max=200)
    ]
    color_handles = [
        Line2D(
            [], [], linestyle="none", marker="o", markersize=7,
            markerfacecolor=cmap(norm(t)), markeredgecolor="white",
            markeredgewidth=0.65, alpha=0.78,
        )
        for t in color_ticks
    ]

    leg_size = ax.legend(
        size_handles, [f"{t:.2g}" for t in size_ticks],
        title=size_label, loc="upper right", bbox_to_anchor=(1.0, 1.0),
        frameon=True, fontsize=8, title_fontsize=8, labelspacing=1.1,
    )
    style_legend(leg_size)
    ax.add_artist(leg_size)

    leg_color = ax.legend(
        color_handles, [f"{t:.2g}" for t in color_ticks],
        title=color_label, loc="upper right", bbox_to_anchor=(1.0, 0.62),
        frameon=True, fontsize=8, title_fontsize=8, labelspacing=1.1,
    )
    style_legend(leg_color)


def plot_dense_bubble(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    size: str,
    color: str,
    xlabel: str,
    ylabel: str,
    size_label: str,
    color_label: str,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8.0, 6.5),
    alpha: float = 0.72,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Dense point-cloud bubble plot (reference style)."""
    apply_plot_style()
    cmap = rose_navy_cmap()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize, facecolor="white", constrained_layout=True)

    style_axes_minimal(ax)
    sizes = scale_bubble_sizes(df[size].values)
    cvals = df[color].astype(float).values
    norm = Normalize(vmin=np.nanmin(cvals), vmax=np.nanmax(cvals))

    ax.scatter(
        df[x], df[y], s=sizes, c=cvals, cmap=cmap, norm=norm,
        alpha=alpha, edgecolors="white", linewidths=0.65,
        rasterized=True, zorder=3,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        panel_title(ax, title)

    _compact_bubble_legends(ax, df[size].values, cvals, size_label, color_label, cmap)

    if save_path:
        paths = save_figure(ax.figure, save_path, formats=("png", "pdf", "svg"))
        print(f"Saved dense bubble chart: {', '.join(paths)}")
    if created and not save_path:
        plt.show()
    return ax


def plot_bubble_chart(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    size: str,
    color: str,
    xerr: Optional[str] = None,
    yerr: Optional[str] = None,
    xlabel: str,
    ylabel: str,
    size_label: str,
    color_label: str,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (7.5, 6.0),
    save_path: Optional[str] = None,
    interactive: bool = False,
) -> plt.Axes:
    """Sparse summary bubble plot (few points, optional error bars)."""
    apply_plot_style()
    cmap = rose_navy_cmap()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize, facecolor="white", constrained_layout=True)

    style_axes_minimal(ax)
    sizes = scale_bubble_sizes(df[size].values, s_min=80, s_max=520)
    cvals = df[color].astype(float).values
    norm = Normalize(vmin=np.nanmin(cvals), vmax=np.nanmax(cvals))

    for idx, (_, row) in enumerate(df.iterrows()):
        if xerr or yerr:
            x_e = float(row[xerr]) if xerr and pd.notna(row[xerr]) else None
            y_e = float(row[yerr]) if yerr and pd.notna(row[yerr]) else None
            if x_e is not None or y_e is not None:
                ax.errorbar(
                    row[x], row[y], xerr=x_e, yerr=y_e,
                    fmt="none", ecolor="#9CA3AF", alpha=0.45, capsize=2.5, zorder=2,
                )
        ax.scatter(
            row[x], row[y], s=sizes[idx],
            c=[cmap(norm(cvals[idx]))], alpha=0.78,
            edgecolors="white", linewidths=1.0, zorder=4,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        panel_title(ax, title)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    _compact_bubble_legends(ax, df[size].values, cvals, size_label, color_label, cmap)

    if save_path:
        paths = save_figure(ax.figure, save_path, formats=("png", "pdf", "svg"))
        print(f"Saved bubble chart: {', '.join(paths)}")

    if interactive:
        _plot_bubble_plotly(df, x, y, size, color, xlabel, ylabel, size_label, color_label, title, save_path)

    if created and not save_path:
        plt.show()
    return ax


def _plot_bubble_plotly(
    df, x, y, size, color, xlabel, ylabel, size_label, color_label, title, save_path,
) -> None:
    try:
        import plotly.express as px
    except ImportError:
        print("Plotly not installed — skipping interactive bubble chart.")
        return
    fig = px.scatter(
        df, x=x, y=y, size=size, color=color,
        size_max=28, opacity=0.75,
        color_continuous_scale=ROSE_NAVY_SEQUENCE,
        labels={x: xlabel, y: ylabel, size: size_label, color: color_label},
        title=title or "",
    )
    fig.update_traces(marker=dict(line=dict(width=0.8, color="white")))
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12, color=TITLE_COLOR),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
    )
    if save_path:
        html_path = Path(save_path).with_suffix(".html")
        fig.write_html(str(html_path))
        print(f"Saved interactive bubble chart: {html_path}")


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 600,
) -> List[str]:
    """Save *fig* as PNG (preview) and PDF (papers); pass ``svg`` in *formats* if needed."""
    base = Path(path)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    facecolor = fig.get_facecolor()
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor=facecolor)
        saved.append(str(out))
    return saved
