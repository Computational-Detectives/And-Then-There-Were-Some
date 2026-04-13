"""
ego_borda_viz.py
================
Visualisations for the longevity-adjusted Borda scoring of ego-network
structural metrics in *And Then There Were None*.

Usage
-----
    ``python ego_borda_viz.py`` \n
        ``--per-window   path/to/borda_per_window.csv``\n
      ``--aggregated   path/to/borda_aggregated.csv``\n
      ``--output-dir   path/to/output``
"""

from __future__ import annotations

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from math import pi
from typing import List, Dict, Tuple, Any


# ------- CONSTANTS -------

SHORT = {
    "Lawrence John Wargrave":    "Wargrave",
    "Vera Elizabeth Claythorne": "Vera",
    "Philip Lombard":            "Lombard",
    "Edward George Armstrong":   "Armstrong",
    "William Henry Blore":       "Blore",
    "Emily Brent":               "Brent",
    "John Gordon Macarthur":     "Macarthur",
    "Thomas Rogers":             "Rogers",
    "Ethel Rogers":              "E.Rogers",
    "Anthony James Marston":     "Marston",
}

VICTIM_BY_WINDOW_LO: Dict[int, str] = {
    0:     "Marston",
    17463: "E.Rogers",
    21071: "Macarthur",
    32905: "Rogers",
    41439: "Brent",
    45358: "Wargrave",
    50821: "Blore",
    58127: "Armstrong",
    59108: "Lombard",
}

_CHAR_ORDER = [
    "Vera", "Armstrong", "Lombard", "Wargrave", "Blore",
    "Brent", "Macarthur", "Rogers", "E.Rogers", "Marston",
]
_CMAP = plt.cm.tab10
CHAR_COLOURS: Dict[str, Tuple[Any]] = {
    c: _CMAP(i / max(len(_CHAR_ORDER) - 1, 1))
    for i, c in enumerate(_CHAR_ORDER)
}


# ============================================================================
# HELPERS
# ============================================================================

def _save(fig: plt.Figure, output_dir: str, filename: str, show: bool) -> None:
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def _short(name: str) -> str:
    return SHORT.get(name, name)


def _window_labels(windows_lo: List[int]) -> List[str]:
    """Return 'W{i} ({victim})' labels for each window_lo."""
    return [
        f"W{i+1}\n({VICTIM_BY_WINDOW_LO.get(w, '?')})"
        for i, w in enumerate(sorted(windows_lo))
    ]


# ============================================================================
# HEATMAP
# ============================================================================

def plot_borda_heatmap(
    pw: pd.DataFrame,
    agg: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """
    Per-window Heatmap of ``borda_composite_3`` contributions.

    A more row that has uniformly high values across many columns points to
    to a character that sutained their structural dominance roughly quantifying 
    the deliberate choice in behaviour of the structural killer. A row that has 
    high values only in early or only in late windows suggests that character's 
    structural position is either influenced by early dominance or mechanical 
    network collapse as fewer and fewer characters are alive.
    """
    pw = pw.copy()
    pw["short_name"] = pw["name"].map(_short)

    windows = sorted(pw["window_lo"].unique())
    col_labels = _window_labels(windows)

    # Sort characters by aggregated composite_3 Borda (desc)
    agg = agg.copy()
    agg["short_name"] = agg["name"].map(_short)
    char_order = (
        agg.sort_values("longevity_adjusted_borda_composite_3", ascending=False)
        ["short_name"].tolist()
    )

    # Build matrix
    import numpy as np
    import pandas as pd
    matrix = pd.DataFrame(
        np.nan, index=char_order, columns=range(len(windows))
    )
    win_idx = {w: i for i, w in enumerate(windows)}
    for _, row in pw.iterrows():
        sn = row["short_name"]
        wi = win_idx.get(row["window_lo"])
        val = row.get("borda_composite_3")
        if sn in matrix.index and wi is not None and pd.notna(val):
            matrix.loc[sn, wi] = float(val)

    fig, ax = plt.subplots(figsize=(13, max(4, len(char_order) * 0.65)))

    cmap = plt.cm.YlOrRd
    cmap.set_bad(color="#d0d0d0")
    masked = np.ma.masked_invalid(matrix.values.astype(float))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Per-window Borda score [0–1]", fontsize=9)

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(char_order)))
    ax.set_yticklabels(char_order, fontsize=9)

    # Cell borders
    for i in range(len(char_order)):
        for j in range(len(windows)):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="white", linewidth=0.8,
            ))
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if val > 0.6 else "black")

    ax.set_xlabel("Window (victim who dies at end)", fontsize=9)
    ax.set_title(
        "Per-window longevity-adjusted Borda score — composite (eff. size, constraint, hierarchy)\n"
        "Rows sorted by total Borda score descending  |  grey = absent / no data",
        fontsize=10,
    )
    plt.tight_layout()
    _save(fig, output_dir, "borda_heatmap.png", show)


# ============================================================================
# BUMP CHART
# ============================================================================

def plot_borda_bump(
    pw: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """
    Cross-window Bump chart over ``rank_composite_3``.

    The y-axis is inverted so rank 1 appears at the top. Each line is one
    character.

    A flat line near rank 1 throughout points to a stable structural
    dominance i.e., they are more consistent with the killer-profile.
    """
    import pandas as pd

    pw = pw.copy()
    pw["short_name"] = pw["name"].map(_short)
    pw["window_idx"] = pd.factorize(pw["window_lo"], sort=True)[0] + 1

    windows = sorted(pw["window_lo"].unique())
    n_windows = len(windows)

    # Victim x-axis labels
    xtick_labels = _window_labels(windows)

    fig, ax = plt.subplots(figsize=(12, 6))

    plotted = {}
    for char, grp in pw.groupby("short_name"):
        sub = grp[grp["rank_composite_3"].notna()].sort_values("window_idx")
        if len(sub) == 0:
            continue
        colour = CHAR_COLOURS.get(char, "#888888")
        ax.plot(
            sub["window_idx"],
            sub["rank_composite_3"],
            marker="o",
            color=colour,
            linewidth=2,
            markersize=6,
            zorder=3,
            label=char,
        )
        plotted[char] = sub

    # Annotations at first and last valid window
    for char, sub in plotted.items():
        colour = CHAR_COLOURS.get(char, "#888888")
        first = sub.iloc[0]
        last  = sub.iloc[-1]
        # Left annotation
        ax.annotate(
            char,
            (first["window_idx"], first["rank_composite_3"]),
            xytext=(-6, 0), textcoords="offset points",
            ha="right", va="center", fontsize=7.5, color=colour,
        )
        # Right annotation only if last window differs from first
        if last["window_idx"] != first["window_idx"]:
            ax.annotate(
                char,
                (last["window_idx"], last["rank_composite_3"]),
                xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.5, color=colour,
            )

    ax.set_xticks(range(1, n_windows + 1))
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylabel("Composite rank (rank 1 = best structural position)", fontsize=9)
    ax.set_xlabel("Window (victim who dies at end)", fontsize=9)
    ax.set_title(
        "Bump chart — composite rank (eff. size, constraint, hierarchy) across windows\n"
        "Rank 1 at top  |  flat lines near top = sustained structural dominance",
        fontsize=10,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(axis="x", linestyle=":", alpha=0.2)
    ax.set_xlim(0.3, n_windows + 0.7)
    plt.tight_layout()
    _save(fig, output_dir, "borda_bump.png", show)


# ============================================================================
# GROUPED BAR CHART
# ============================================================================

def plot_borda_grouped_bar(
    agg: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """
    Grouped bar chart of ``avg_borda_per_window`` per metric per character.

    Four metric groups: ``effective_size, constraint, hierarchy, density_moderate``.
    Each group contains one bar per character.  The ``median_borda_per_window``
    for each metric is overlaid as a horizontal tick mark on each bar,
    making it easy to assess whether the average is driven up by outlier
    windows or is genuinely sustained.

    Characters are sorted by ``avg_borda_per_window_composite_3`` descending
    so the most structurally fitting character appears first w.r.t. killer-profile.

    A character with uniformly tall bars across all four
    metrics has structural-profile consistency with the killer-profile.
    A character's profile with one tall bar and three short bars is dominated
    by a single-metric.
    """
    import pandas as pd

    agg = agg.copy()
    agg["short_name"] = agg["name"].map(_short)
    agg = agg.sort_values("avg_borda_per_window_composite_3", ascending=False)

    metrics = ["effective_size", "constraint", "hierarchy", "density_moderate"]
    metric_labels = ["Eff. size", "Constraint", "Hierarchy", "Density\n(moderate)"]
    n_metrics = len(metrics)
    n_chars   = len(agg)

    bar_width = 0.18
    group_gap = 0.25
    group_width = n_chars * bar_width

    fig, ax = plt.subplots(figsize=(13, 6))

    for gi, (metric, label) in enumerate(zip(metrics, metric_labels)):
        avg_col = f"avg_borda_per_window_{metric}"
        med_col = f"median_borda_per_window_{metric}"

        for ci, (_, row) in enumerate(agg.iterrows()):
            char = row["short_name"]
            x = gi * (group_width + group_gap) + ci * bar_width

            avg_val = row.get(avg_col)
            med_val = row.get(med_col)
            colour  = CHAR_COLOURS.get(char, "#888888")

            if pd.notna(avg_val):
                ax.bar(x, float(avg_val), width=bar_width * 0.85,
                       color=colour, alpha=0.8, zorder=3)

            if pd.notna(med_val) and pd.notna(avg_val):
                # Median as horizontal tick across bar
                ax.plot(
                    [x - bar_width * 0.35, x + bar_width * 0.35],
                    [float(med_val), float(med_val)],
                    color="black", linewidth=1.6, zorder=4,
                )

    # Group x-axis labels (centred under each group)
    group_centres = [
        gi * (group_width + group_gap) + group_width / 2 - bar_width / 2
        for gi in range(n_metrics)
    ]
    ax.set_xticks(group_centres)
    ax.set_xticklabels(metric_labels, fontsize=10)

    # Character legend
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                       color=CHAR_COLOURS.get(c, "#888"), alpha=0.8,
                       label=c)
        for c in agg["short_name"].tolist()
    ]
    ax.legend(handles=handles, fontsize=7.5, ncol=5,
              loc="upper right", framealpha=0.9)

    # Median legend entry
    ax.plot([], [], color="black", linewidth=1.6, label="median")
    ax.legend(handles=handles + [
        plt.Line2D([0], [0], color="black", linewidth=1.6, label="median")
    ], fontsize=7.5, ncol=5, loc="upper right", framealpha=0.9)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Avg. per-window Borda score [0-1]", fontsize=9)
    ax.set_title(
        "Average per-window Borda score by metric and character\n"
        "Bars = avg  |  horizontal tick = median  |  "
        "sorted by composite (eff. size + constraint + hierarchy)",
        fontsize=10,
    )
    ax.axhline(0.5, linestyle="--", color="grey", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save(fig, output_dir, "borda_grouped_bar.png", show)


# ============================================================================
# RADAR - PLOTS PROTAGONISTS BEHAVIOURAL PROFILE
# ============================================================================

def plot_borda_radar(
    agg: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """Small-multiple radar charts — one per character.

    Each radar has five axes:
        ``effective_size, constraint, hierarchy, density_moderate, composite_3``

    Values are the ``longevity_adjusted_borda`` for each axis, normalised to
    [0, 1] by dividing by the maximum observed value across all characters
    on that axis. This puts all characters on a common radial scale.

    A large, balanced polygon indicates strong and broad structural
    positioning more consistent with the full killer profile. A small polygon
    indicates overall weak positioning. A lopsided polygon indicates
    single-metric dependence. The ``composite_3`` axis serves as a summary
    in whcih characters who score well on individual metrics but poorly on
    the composite have conflicting metric directions within windows.

    Characters are arranged in order of descending ``longevity_adjusted_borda
    _composite_3`` (most suspicious first, reading left-to-right, top-to-bottom).
    """
    import numpy as np
    import pandas as pd

    agg = agg.copy()
    agg["short_name"] = agg["name"].map(_short)
    agg = agg.sort_values("longevity_adjusted_borda_composite_3", ascending=False)

    axes_keys = [
        "longevity_adjusted_borda_effective_size",
        "longevity_adjusted_borda_constraint",
        "longevity_adjusted_borda_hierarchy",
        "longevity_adjusted_borda_density_moderate",
        "longevity_adjusted_borda_composite_3",
    ]
    axes_labels = [
        "Eff. size",
        "Constraint",
        "Hierarchy",
        "Density\n(mod.)",
        "Composite",
    ]
    N = len(axes_keys)
    angles = [2 * pi * i / N for i in range(N)] + [2 * pi * 0 / N]  # close polygon

    # Normalise each axis to [0, 1]
    maxima = {k: agg[k].max() for k in axes_keys if k in agg.columns}
    maxima = {k: v if v > 0 else 1.0 for k, v in maxima.items()}

    n_chars = len(agg)
    ncols = 4
    nrows = int(np.ceil(n_chars / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 3.2),
        subplot_kw=dict(polar=True),
    )
    axes_flat = np.array(axes).flatten()

    for i, (_, row) in enumerate(agg.iterrows()):
        ax = axes_flat[i]
        char = row["short_name"]
        colour = CHAR_COLOURS.get(char, "#888888")

        vals = []
        for k in axes_keys:
            raw = row.get(k)
            if pd.isna(raw):
                vals.append(0.0)
            else:
                vals.append(float(raw) / maxima.get(k, 1.0))
        vals_closed = vals + [vals[0]]

        ax.plot(angles, vals_closed, color=colour, linewidth=1.8, zorder=3)
        ax.fill(angles, vals_closed, color=colour, alpha=0.22)

        # Axis labels
        ax.set_thetagrids(
            [a * 180 / pi for a in angles[:-1]],
            axes_labels,
            fontsize=7,
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""], fontsize=0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.grid(True, linestyle="--", alpha=0.3)

        windows_n = int(row.get("windows_present", 0))
        la_borda  = row.get("longevity_adjusted_borda_composite_3")
        subtitle  = f"n={windows_n}  |  Σ={la_borda:.2f}" if pd.notna(la_borda) else f"n={windows_n}"
        ax.set_title(f"{char}\n{subtitle}", fontsize=8.5, pad=10, color=colour)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Borda decomposition radar — per-metric longevity-adjusted Borda (normalised to [0-1])\n"
        "Sorted by composite score descending  |  n = windows present  |  Σ = composite Borda total",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    _save(fig, output_dir, "borda_radar.png", show)


# ============================================================================
# VISUALISE BORDA RANKINGS
# ============================================================================

def run(
    per_window_path: str,
    aggregated_path: str,
    output_dir: str,
) -> None:
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    pw  = pd.read_csv(per_window_path)
    agg = pd.read_csv(aggregated_path)
    print(f"  Per-window: {len(pw)} rows  |  Aggregated: {len(agg)} characters")

    print("1/4 — Borda heatmap...")
    plot_borda_heatmap(pw, agg, output_dir)

    print("2/4 — Bump chart...")
    plot_borda_bump(pw, output_dir)

    print("3/4 — Grouped bar chart...")
    plot_borda_grouped_bar(agg, output_dir)

    print("4/4 — Radar decomposition...")
    plot_borda_radar(agg, output_dir)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Borda score visualisations for ego-network killer profiling.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--per-window", required=True,
        help="Path to borda_per_window.csv",
    )
    parser.add_argument(
        "--aggregated", required=True,
        help="Path to borda_aggregated.csv",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for output PNG files",
    )

    args = parser.parse_args()

    # Visualise Borda Ranking
    run(
        per_window_path=args.per_window,
        aggregated_path=args.aggregated,
        output_dir=args.output_dir,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
