"""
Visualisation helpers for character-timeline analysis.

Functions:
    _draw_death_shading       – Vertical bands marking character deaths
    plot_heatmap              – Character mention density heatmap
    plot_streamgraph          – Stacked area chart of mentions
    plot_lines                – Individual line chart per character
    plot_lines_with_derivative – Two-panel: raw mentions + smoothed derivative
    plot_hierarchy_timeline   – Per-character Burt hierarchy over time
"""
from __future__ import annotations

import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...config import DEATH_INTERVALS


# ─────────────────────────────────────────────
# DEATH SHADING HELPER
# ─────────────────────────────────────────────

def _draw_death_shading(
    ax, edges, n_bins, sid_min: int, sid_max: int, is_heatmap: bool = False,
) -> None:
    """Draw semi-transparent vertical bands for each character death."""
    sid_span = max(sid_max - sid_min, 1)

    for _i, (label, lo, hi) in enumerate(DEATH_INTERVALS):
        # ── Map death sentence IDs → x-axis coordinates ──
        if is_heatmap or edges is None:
            # Heatmap or chapter-mode line charts: x = bin index
            x_lo = (lo - sid_min) / sid_span * (n_bins - 1)
            x_hi = (hi - sid_min) / sid_span * (n_bins - 1)
        else:
            # Stream / line charts with edge-derived x-axis
            e_min, e_max = float(edges[0]), float(edges[-1])
            x_lo = e_min + (lo - sid_min) / sid_span * (e_max - e_min)
            x_hi = e_min + (hi - sid_min) / sid_span * (e_max - e_min)

        # Clip to visible range
        xlim = ax.get_xlim()
        x_lo = max(x_lo, xlim[0])
        x_hi = min(x_hi, xlim[1])
        if x_hi <= x_lo:
            continue

        ax.axvspan(x_lo, x_hi, alpha=0.10, color="firebrick", zorder=0)
        mid_x = (x_lo + x_hi) / 2
        ylim = ax.get_ylim()
        y_top = ylim[1] - (ylim[1] - ylim[0]) * 0.02
        ax.text(
            mid_x, y_top, label,
            ha="center", va="top", fontsize=5, rotation=90,
            alpha=0.55, color="firebrick",
        )


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    tick_pos, tick_lab,
    xlabel: str,
    out_dir: Path,
    suffix: str = "",
    sid_range: tuple[int, int] | None = None,
    n_bins_total: int | None = None,
) -> None:
    _, ax = plt.subplots(figsize=(16, max(5, len(labels) * 0.5 + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="inferno", interpolation="nearest")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Character", fontsize=11)
    ax.set_title("Character Mention Density Across the Narrative",
                 fontsize=13, fontweight="bold", pad=12)

    if sid_range:
        nb = n_bins_total or matrix.shape[1]
        _draw_death_shading(ax, None, nb, sid_range[0], sid_range[1], is_heatmap=True)

    plt.colorbar(im, ax=ax, label="Mentions per bin", shrink=0.8)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"character_timeline_heatmap{suffix}.png"
    plt.savefig(out_dir / fname, dpi=180)
    plt.close()
    print(f"  Saved → {out_dir / fname}")


def plot_streamgraph(
    matrix: np.ndarray,
    sorted_names: list[str],
    n_bins: int,
    edges,
    xlabel: str,
    out_dir: Path,
    suffix: str = "",
    tick_pos=None,
    tick_lab=None,
    kind: str = "",
    sid_range: tuple[int, int] | None = None,
) -> None:
    # X-axis centres
    if edges is not None:
        bin_centres = 0.5 * (edges[:-1] + edges[1:])
    else:
        bin_centres = np.arange(n_bins)

    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(sorted_names))
    colours = [cmap(i) for i in range(len(sorted_names))]

    _, ax = plt.subplots(figsize=(16, 6))
    ax.stackplot(bin_centres, matrix, labels=sorted_names, colors=colours, alpha=0.85)

    # Apply explicit ticks when provided (e.g. chapter labels)
    if tick_pos is not None and tick_lab is not None:
        ax.set_xticks(bin_centres[tick_pos] if len(bin_centres) > 0 else tick_pos)
        ax.set_xticklabels(tick_lab, rotation=45, ha="right", fontsize=8)

    if sid_range:
        _draw_death_shading(ax, edges, n_bins, sid_range[0], sid_range[1])

    ax.set_xlabel(xlabel, fontsize=11)
    y_label = "Co-occurrence Mentions" if kind == 'cooccurrence' else "Mentions"
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title("Narrative Focus by Character Over Time",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = f"character_timeline_stream{suffix}.png"
    plt.savefig(out_dir / fname, dpi=180)
    plt.close()
    print(f"  Saved → {out_dir / fname}")


def plot_lines(
    matrix: np.ndarray,
    sorted_names: list[str],
    n_bins: int,
    edges,
    xlabel: str,
    out_dir: Path,
    suffix: str = "",
    tick_pos=None,
    tick_lab=None,
    kind: str = "",
    sid_range: tuple[int, int] | None = None,
) -> None:
    """Individual lines per character, all on one axis."""
    if edges is not None:
        bin_centres = 0.5 * (edges[:-1] + edges[1:])
    else:
        bin_centres = np.arange(n_bins)

    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(sorted_names))

    _, ax = plt.subplots(figsize=(16, 6))
    for i, name in enumerate(sorted_names):
        ax.plot(bin_centres, matrix[i], label=name, color=cmap(i), linewidth=1.4, alpha=0.85)

    # Apply explicit ticks when provided (e.g. chapter labels)
    if tick_pos is not None and tick_lab is not None:
        ax.set_xticks(bin_centres[tick_pos] if len(bin_centres) > 0 else tick_pos)
        ax.set_xticklabels(tick_lab, rotation=45, ha="right", fontsize=8)

    if sid_range:
        _draw_death_shading(ax, edges, n_bins, sid_range[0], sid_range[1])

    ax.set_xlabel(xlabel, fontsize=11)
    y_label = "Co-occurrence Mentions" if kind == 'cooccurrence' else "Mentions"
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title("Per-Character Mentions Over Time",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = f"character_timeline_lines{suffix}.png"
    plt.savefig(out_dir / fname, dpi=180)
    plt.close()
    print(f"  Saved → {out_dir / fname}")


def plot_lines_with_derivative(
    matrix: np.ndarray,
    sorted_names: list[str],
    n_bins: int,
    edges,
    xlabel: str,
    out_dir: Path,
    suffix: str = "",
    tick_pos=None,
    tick_lab=None,
    kind: str = "",
    sigma: float = 1.5,
    sid_range: tuple[int, int] | None = None,
) -> None:
    """
    Two-panel subplot: raw mentions (top, solid) and smoothed first
    derivative (bottom, dashed).  Gaussian smoothing with `sigma` bins
    is applied before differentiation to suppress noise.
    """
    from scipy.ndimage import gaussian_filter1d

    if edges is not None:
        bin_centres = 0.5 * (edges[:-1] + edges[1:])
    else:
        bin_centres = np.arange(n_bins)

    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(sorted_names))

    _, (ax_raw, ax_deriv) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.12},
        layout="constrained",
    )

    for i, name in enumerate(sorted_names):
        colour = cmap(i)
        raw = matrix[i].astype(float)

        # Smooth then differentiate
        smoothed = gaussian_filter1d(raw, sigma=sigma)
        deriv = np.gradient(smoothed)

        ax_raw.plot(bin_centres, raw, label=name, color=colour,
                    linewidth=1.4, alpha=0.85)
        ax_deriv.plot(bin_centres, deriv, label=name, color=colour,
                      linewidth=1.2, alpha=0.8, linestyle="--")

    # Zero reference line on the derivative panel
    ax_deriv.axhline(0, color="grey", linewidth=0.6, linestyle=":")

    # Apply explicit ticks when provided (e.g. chapter labels)
    if tick_pos is not None and tick_lab is not None:
        ticks = bin_centres[tick_pos] if len(bin_centres) > 0 else tick_pos
        ax_deriv.set_xticks(ticks)
        ax_deriv.set_xticklabels(tick_lab, rotation=45, ha="right", fontsize=8)

    if sid_range:
        _draw_death_shading(ax_raw, edges, n_bins, sid_range[0], sid_range[1])
        _draw_death_shading(ax_deriv, edges, n_bins, sid_range[0], sid_range[1])

    y_label = "Co-occurrence Mentions" if kind == "cooccurrence" else "Mentions"
    ax_raw.set_ylabel(y_label, fontsize=11)
    ax_raw.set_title("Per-Character Mentions & Rate of Change",
                     fontsize=13, fontweight="bold", pad=12)
    ax_raw.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    ax_deriv.set_xlabel(xlabel, fontsize=11)
    ax_deriv.set_ylabel("Δ Mentions / Δ bin", fontsize=11)
    ax_deriv.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)

    os.makedirs(out_dir, exist_ok=True)
    fname = f"character_timeline_derivative{suffix}.png"
    plt.savefig(out_dir / fname, dpi=180)
    plt.close()
    print(f"  Saved → {out_dir / fname}")


def plot_hierarchy_timeline(
    hier_matrix: np.ndarray,
    sorted_names: list[str],
    edges,
    tick_pos,
    tick_lab,
    xlabel: str,
    out_dir: Path,
    suffix: str = "",
    sid_range: tuple[int, int] | None = None,
) -> None:
    """Plot per-character hierarchy values over time as a line chart."""
    n_bins = hier_matrix.shape[1]
    if edges is not None:
        bin_centres = 0.5 * (edges[:-1] + edges[1:])
    else:
        bin_centres = np.arange(n_bins)

    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(sorted_names))

    _, ax = plt.subplots(figsize=(16, 6))
    for i, name in enumerate(sorted_names):
        vals = hier_matrix[i]
        # Only plot segments where hierarchy is defined
        ax.plot(bin_centres, vals, label=name, color=cmap(i),
                linewidth=1.4, alpha=0.85, marker=".", markersize=3)

    if tick_pos is not None and tick_lab is not None:
        ticks = bin_centres[tick_pos] if len(bin_centres) > 0 else tick_pos
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_lab, rotation=45, ha="right", fontsize=8)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Hierarchy (Burt)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

    if sid_range:
        _draw_death_shading(ax, edges, n_bins, sid_range[0], sid_range[1])

    ax.set_title("Ego Hierarchy Over Time",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = f"character_hierarchy_timeline{suffix}.png"
    plt.savefig(out_dir / fname, dpi=180)
    plt.close()
    print(f"  Saved → {out_dir / fname}")
