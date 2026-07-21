#!/usr/bin/env python3
"""Publication box plot of ReLaFIS Antecedent Relation Entropy (ARE).

Raw ARE values come from 30 independent GIFTSHIFTER / GIFTSHIFT runs; the
paper label is ReLaFIS. Matches ``utils.paper_plot_style`` typography and
export settings used by other manuscript figures.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paper_plot_style import (  # noqa: E402
    ANNOTATION_SIZE,
    DOUBLE_COLUMN_WIDTH,
    LEGEND_SIZE,
    apply_paper_style,
    make_legend_compact,
    style_axis,
)
from utils.plot_style import ELEGANT  # noqa: E402

# Dataset order required by the manuscript.
DATASET_ORDER: Sequence[str] = ("HAB", "CRI", "IMU", "IRI", "THY")

# Expected mean ± population std (ddof=0) for validation.
EXPECTED_MEAN_STD: Dict[str, tuple] = {
    "HAB": (0.6990, 0.2047),
    "CRI": (0.6843, 0.1346),
    "IMU": (0.6917, 0.1304),
    "IRI": (0.5281, 0.1434),
    "THY": (1.0056, 0.1629),
}

# Raw ARE (nats) from 30 independent ReLaFIS runs per dataset.
are_results: Dict[str, List[float]] = {
    "HAB": [
        0.6365, 0.8676, 0.6365, 0.6365, 0.3183,
        0.6365, 0.6365, 0.8676, 0.6365, 0.3183,
        0.8676, 0.8676, 0.6365, 1.0986, 0.8676,
        0.6365, 0.8676, 0.6365, 0.5493, 0.8676,
        0.3183, 0.3183, 1.0986, 0.8676, 0.6365,
        0.6365, 0.6365, 0.8676, 0.6365, 0.8676,
    ],
    "CRI": [
        0.5435, 0.6365, 0.6591, 0.6648, 0.8523,
        0.9831, 0.3466, 0.6365, 0.6648, 0.5435,
        0.8240, 0.8676, 0.8240, 0.7520, 0.6365,
        0.6365, 0.8240, 0.7310, 0.6365, 0.7520,
        0.7804, 0.6365, 0.3183, 0.6365, 0.6648,
        0.7520, 0.7310, 0.6648, 0.6648, 0.6648,
    ],
    "IMU": [
        0.6406, 0.6032, 0.2991, 0.8193, 0.7770,
        0.8013, 0.7770, 0.5983, 0.8193, 0.8436,
        0.7770, 0.6406, 0.8193, 0.5465, 0.8436,
        0.6973, 0.6406, 0.6406, 0.6829, 0.6406,
        0.6406, 0.8193, 0.5465, 0.8193, 0.8436,
        0.6829, 0.5021, 0.8436, 0.5042, 0.6406,
    ],
    "IRI": [
        0.6931, 0.6277, 0.5623, 0.5623, 0.5623,
        0.6277, 0.6277, 0.3466, 0.6277, 0.2812,
        0.6277, 0.6277, 0.6931, 0.2812, 0.6277,
        0.6931, 0.3466, 0.5623, 0.6277, 0.6277,
        0.3466, 0.6277, 0.2812, 0.5623, 0.6277,
        0.5623, 0.3466, 0.3466, 0.2812, 0.6277,
    ],
    "THY": [
        1.0549, 0.7777, 1.1935, 1.0026, 1.0549,
        1.3322, 0.8116, 1.1412, 1.0549, 1.1935,
        1.1412, 0.5867, 1.0026, 0.8116, 0.9503,
        0.8640, 0.9503, 1.0026, 1.1935, 0.8640,
        0.8640, 1.0549, 1.1935, 0.8640, 1.1935,
        0.8640, 0.8640, 1.1412, 1.1412, 1.0026,
    ],
}

# Restrained academic palette (deep blue family from project style).
BOX_FACE = "#9BB8D3"
BOX_EDGE = ELEGANT["deep_blue"]
MEDIAN_COLOR = "#1A3558"
POINT_COLOR = ELEGANT["navy"]
MEAN_COLOR = ELEGANT["accent"]
REF_LINE_COLOR = "#64748B"

LN4 = float(np.log(4))  # ≈ 1.38629436112
ARE_TOL = 1e-9
STATS_TOL = 5e-4  # expected mean±std are rounded to 4 decimals


def validate_and_summarize(results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Assert count / range constraints and print summary statistics."""
    summaries: Dict[str, Dict[str, float]] = {}
    print("Dataset   n   mean     std(pop)  median   Q1       Q3")
    print("-" * 64)
    for name in DATASET_ORDER:
        if name not in results:
            raise KeyError(f"Missing dataset in are_results: {name}")
        values = np.asarray(results[name], dtype=float)
        # Treat -0.0 as ordinary 0.0.
        values = np.where(np.isclose(values, 0.0), 0.0, values)

        if values.size != 30:
            raise AssertionError(
                f"{name}: expected 30 ARE values, got {values.size}"
            )
        if np.any(values < -ARE_TOL) or np.any(values > LN4 + ARE_TOL):
            bad = values[(values < -ARE_TOL) | (values > LN4 + ARE_TOL)]
            raise AssertionError(
                f"{name}: ARE values outside [0, ln(4)]: {bad}"
            )

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        median = float(np.median(values))
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        summaries[name] = {
            "mean": mean, "std": std, "median": median, "q1": q1, "q3": q3,
        }
        print(
            f"{name:8s} {values.size:2d}  {mean:.4f}   {std:.4f}    "
            f"{median:.4f}   {q1:.4f}   {q3:.4f}"
        )

        exp_mean, exp_std = EXPECTED_MEAN_STD[name]
        if abs(mean - exp_mean) > STATS_TOL or abs(std - exp_std) > STATS_TOL:
            raise AssertionError(
                f"{name}: calculated mean±std {mean:.4f} ± {std:.4f} "
                f"differs from expected {exp_mean:.4f} ± {exp_std:.4f}"
            )
    print("-" * 64)
    print("Validation OK (mean ± pop. std match expected values within 5e-4).")
    return summaries


def plot_are_boxplot(
    results: Dict[str, List[float]],
    *,
    save_path: Path | str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Draw Tukey box plots with jittered individual-run markers."""
    apply_paper_style()
    summaries = validate_and_summarize(results)

    data = [np.asarray(results[name], dtype=float) for name in DATASET_ORDER]
    positions = np.arange(1, len(DATASET_ORDER) + 1)

    # Readable at one- and two-column paper widths.
    # Use subplots_adjust (not constrained_layout) so the left margin can
    # keep the y-axis label clear of the exported figure edge.
    fig_w = min(DOUBLE_COLUMN_WIDTH, 6.2)
    fig, ax = plt.subplots(figsize=(fig_w, 3.45), facecolor="white")
    style_axis(ax, grid=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.55, alpha=0.40, color="#CBD5E1")
    ax.xaxis.grid(False)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        notch=False,
        whis=1.5,
        # All 30 runs are overlaid below; avoid duplicate outlier markers.
        showfliers=False,
        boxprops=dict(linewidth=1.05, edgecolor=BOX_EDGE, facecolor=BOX_FACE, alpha=0.85),
        whiskerprops=dict(linewidth=1.0, color=BOX_EDGE),
        capprops=dict(linewidth=1.0, color=BOX_EDGE),
        medianprops=dict(linewidth=1.85, color=MEDIAN_COLOR, solid_capstyle="butt"),
        zorder=3,
    )
    # Emphasise median slightly more than the box border.
    for med in box["medians"]:
        med.set_zorder(4)

    rng = np.random.default_rng(42)
    means = []
    for pos, values in zip(positions, data):
        jitter = rng.uniform(-0.14, 0.14, size=len(values))
        ax.scatter(
            pos + jitter,
            values,
            s=14,
            c=POINT_COLOR,
            alpha=0.38,
            edgecolors="none",
            zorder=5,
            clip_on=True,
        )
        means.append(summaries[DATASET_ORDER[pos - 1]]["mean"])

    ax.scatter(
        positions,
        means,
        marker="D",
        s=28,
        c=MEAN_COLOR,
        edgecolors="white",
        linewidths=0.55,
        zorder=6,
        label="Mean",
    )

    ax.axhline(
        LN4, color=REF_LINE_COLOR, linestyle="--", linewidth=1.0, alpha=0.85, zorder=2,
    )
    ax.text(
        positions[-1] + 0.42,
        LN4,
        r"$\ln(4)$",
        va="center",
        ha="left",
        fontsize=ANNOTATION_SIZE,
        color=REF_LINE_COLOR,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(list(DATASET_ORDER))
    ax.set_xlabel("Dataset")
    ax.set_ylabel(
        "ARE (nats)",
        fontsize=13,
        fontweight="semibold",
        labelpad=10,
    )
    ax.set_xlim(0.4, len(DATASET_ORDER) + 0.85)
    ax.set_ylim(0.0, 1.42)
    # No internal title — caption carries the description (paper convention).
    fig.subplots_adjust(left=0.15)

    legend_handles = [
        Line2D(
            [0], [0], linestyle="none", marker="o", markersize=5,
            markerfacecolor=POINT_COLOR, markeredgecolor="none", alpha=0.55,
            label="Individual runs",
        ),
        Line2D(
            [0], [0], linestyle="none", marker="D", markersize=5.5,
            markerfacecolor=MEAN_COLOR, markeredgecolor="white",
            markeredgewidth=0.5, label="Mean",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower right",
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.3,
        handlelength=1.2,
        handletextpad=0.4,
        fontsize=LEGEND_SIZE,
    )
    make_legend_compact(legend)

    if save_path is not None:
        paths = _save_all_formats(fig, Path(save_path))
        print("Saved:", ", ".join(paths))

    if show:
        plt.show()
    return fig


def _save_all_formats(fig: plt.Figure, base: Path) -> List[str]:
    """PNG (≥600 dpi) plus vector PDF/SVG with embedded fonts."""
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    facecolor = fig.get_facecolor()
    for fmt in ("png", "pdf", "svg"):
        out = base.with_suffix(f".{fmt}")
        fig.savefig(
            out,
            dpi=600 if fmt == "png" else None,
            bbox_inches="tight",
            pad_inches=0.10,
            facecolor=facecolor,
        )
        saved.append(str(out))
    return saved


def main() -> None:
    out = ROOT / "figures" / "relafis_are_boxplot"
    fig = plot_are_boxplot(are_results, save_path=out, show=False)
    plt.close(fig)

    for fmt in ("png", "pdf", "svg"):
        path = out.with_suffix(f".{fmt}")
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty export: {path}")
        print(f"OK {path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
