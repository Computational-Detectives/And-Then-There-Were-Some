"""
ego_borda.py
============
Per-metric and composite Borda scoring for results of
ego-network analysis in `ego_network.py`.

Usage
-----
    ``python ego_borda.py`` \n
        ``--per-window  path/to/ego_ranks_per_window.csv``\n
      ``--output-dir  path/to/output``

Outputs
-------
    borda_per_window.csv    — per-character per-window rank and Borda contribution
                              for each metric variant
    borda_aggregated.csv    — per-character aggregated Borda scores, sorted by
                              composite longevity-adjusted Borda descending
"""

from __future__ import annotations

import os
import math
import argparse

from pathlib import Path
from typing import List, Dict, Tuple


# ------- CONSTANTS -------

# (column, ascending, label)
METRIC_RANK_DEFS: List[Tuple[str, bool, str]] = [
    ("effective_size", False, "effective_size"),    # descending effective_size = higher is better
    ("constraint",     True,  "constraint"),        # ascending constraint = lower is better
    ("hierarchy",      True,  "hierarchy"),         # ascending hierarchy = lower is better
    ("density",        True,  "density_raw"),       # ascending raw = lower is better
]

# Moderate-density target (midpoint of killer-profile zone 0.2–0.4)
DENSITY_MODERATE_TARGET = 0.3

# Composite definitions: list of rank column names that form each composite
COMPOSITES: Dict[str, List[str]] = {
    # Three-metric killer profile composite (no density)
    "composite_3":  ["rank_effective_size", "rank_constraint", "rank_hierarchy"],
    # Four-metric extended composite (includes density moderate)
    "composite_4":  ["rank_effective_size", "rank_constraint",
                     "rank_hierarchy",      "rank_density_moderate"],
}


# ============================================================================
# LOADING
# ============================================================================

def load_per_window(path: str) -> pd.DataFrame:
    """
    Load the stacked per-window ego_metrics CSV.

    Accepts either a direct path to a stacked CSV or a directory containing
    per-window ego_metrics.csv files (stacks them automatically).
    """
    import pandas as pd
    p = Path(path)
    if p.is_dir():
        frames = []
        for f in sorted(p.rglob("ego_metrics.csv")):
            frames.append(pd.read_csv(f))
        if not frames:
            raise FileNotFoundError(f"No ego_metrics.csv found under {path}")
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(path)


def stack_window_csvs(root: Path) -> pd.DataFrame:
    import pandas as pd
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(root.rglob("ego_metrics.csv")):
        df = pd.read_csv(csv_path)
        if "window_lo" not in df.columns or df["window_lo"].isna().all():
            parts = csv_path.parent.name.split("_")
            if len(parts) == 3 and parts[0] == "window":
                df["window_lo"] = int(parts[1])
                df["window_hi"] = int(parts[2])
            else:
                df["window_lo"] = None
                df["window_hi"] = None
        if "n_alive" not in df.columns:
            df["n_alive"] = None
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No ego_metrics.csv files found under {root}")
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# WITHIN-WINDOW BORDA RANKING
# ============================================================================

def add_per_metric_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add within-window rank columns for all four metrics plus density moderate.

    For each window, characters with missing metric values receive the worst
    rank (``na_option='bottom'``) so that absent characters do not bias the
    rankings of present ones.

    Note
    ----
    Density is the only metric where "moderate" rather than "extreme" is the
    killer-consistent direction.  Two operationalisations are provided:

        rank_density_raw      : rank by raw density value, ascending (lower = better,
                                following the general brokerage interpretation
                                that sparser alter-alter ties equate to more structural holes)

        rank_density_moderate : rank by |density - 0.3| ascending (rank 1 = closest
                                to the moderate zone centre).  This directly
                                tests for Behaviour 1 of the structural killer
                                profile i.e., to have managed moderate density.
    """
    df = df.copy()
    window_groups = ["window_lo", "window_hi"]

    # Standard metrics
    for col, ascending, label in METRIC_RANK_DEFS:
        rank_col = f"rank_{label}"
        if col not in df.columns:
            df[rank_col] = float("nan")
            continue
        df[rank_col] = df.groupby(window_groups)[col].rank(
            ascending=ascending, method="average", na_option="bottom"
        )

    # Density moderate: rank by |density - target|, lower distance = rank 1
    if "density" in df.columns:
        df["_density_dist"] = (df["density"] - DENSITY_MODERATE_TARGET).abs()
        df["rank_density_moderate"] = df.groupby(window_groups)["_density_dist"].rank(
            ascending=True, method="average", na_option="bottom"
        )
        df.drop(columns=["_density_dist"], inplace=True)
    else:
        df["rank_density_moderate"] = float("nan")

    # Composite ranks: mean of constituent per-metric ranks
    for comp_label, rank_cols in COMPOSITES.items():
        available = [c for c in rank_cols if c in df.columns]
        if available:
            df[f"rank_{comp_label}"] = df[available].mean(axis=1)
        else:
            df[f"rank_{comp_label}"] = float("nan")

    return df


# ============================================================================
# PER-WINDOW BORDA CONTRIBUTION
# ============================================================================

def add_borda_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the longevity-adjusted Borda contribution for each character-window.

    For each rank column, the per-window contribution is:

        ``borda_{metric} = max(0, (k - r) / (k - 1))``,

    where ``k = n_alive`` in that window and r = the character's rank.
    Values are in ``[0, 1]: 1.0`` for rank 1, ``0.0`` for last place.

    When ``k == 1`` (i.e., only one survivor) the formula is undefined and the contribution
    is set to ``1.0`` (i.e., sole survivor gets full credit) for composite columns
    and ``NaN`` for individual metric columns (i.e., no competitive ranking possible).
    """
    import pandas as pd
    df = df.copy()

    all_rank_labels = (
        [label for _, _, label in METRIC_RANK_DEFS]
        + ["density_moderate"]
        + [f"composite_{n}" for n in ["3", "4"]]
    )

    for label in all_rank_labels:
        rank_col  = f"rank_{label}"
        borda_col = f"borda_{label}"
        df[borda_col] = float("nan")

        if rank_col not in df.columns:
            continue

        for idx, row in df.iterrows():
            k = row.get("n_alive")
            r = row.get(rank_col)
            if pd.isna(k) or pd.isna(r):
                continue
            k = float(k)
            r = float(r)
            if k <= 1:
                # Single survivor: no ranking possible for individual metrics,
                # full credit for composites (they survived up to this window)
                if "composite" in label:
                    df.at[idx, borda_col] = 1.0
                # else leave as NaN
                continue
            df.at[idx, borda_col] = max(0.0, (k - r) / (k - 1))

    return df


# ============================================================================
# CROSS-WINDOW BORDA AGGREGATION
# ============================================================================

def _top_quartile_pct(grp: pd.DataFrame, rank_col: str) -> float:
    """Percentage of windows where `rank <= ceil(n_alive * 0.25)`."""
    import pandas as pd
    count = 0
    total = 0
    for _, row in grp.iterrows():
        k = row.get("n_alive")
        r = row.get(rank_col)
        if pd.isna(k) or pd.isna(r):
            continue
        total += 1
        if r <= math.ceil(k * 0.25):
            count += 1
    return round(100 * count / total, 1) if total > 0 else float("nan")


def aggregate_borda(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-window Borda contributions into one row per character.

    For each Borda variant computes:
        `longevity_adjusted_borda_{metric}`  — sum of per-window contributions
        `avg_borda_per_window_{metric}`      — sum / windows with valid contribution
        `median_borda_per_window_{metric}`   — median of per-window contributions
        `pct_top_quartile_{metric}`          — % windows in top quartile on rank

    Also computes an overall `composite_borda_rank` (rank among characters on
    `borda_composite_3`, the primary three-metric killer-profile composite) and
    a `borda_composite_4_rank` for the four-metric version including density.

    Returns DataFrame sorted by `longevity_adjusted_borda_composite_3` descending.
    """
    all_labels = (
        [label for _, _, label in METRIC_RANK_DEFS]
        + ["density_moderate"]
        + ["composite_3", "composite_4"]
    )

    records = []

    for (cid, name), grp in df.groupby(["canonical_id", "name"]):
        grp = grp.sort_values("window_lo")
        windows_present = grp["window_lo"].nunique()

        row: dict = {
            "canonical_id":    cid,
            "name":            name,
            "windows_present": windows_present,
        }

        for label in all_labels:
            borda_col = f"borda_{label}"
            rank_col  = f"rank_{label}"

            if borda_col not in grp.columns:
                row[f"longevity_adjusted_borda_{label}"] = float("nan")
                row[f"avg_borda_per_window_{label}"]     = float("nan")
                row[f"median_borda_per_window_{label}"]  = float("nan")
                row[f"pct_top_quartile_{label}"]         = float("nan")
                continue

            valid_borda = grp[borda_col].dropna()

            la_borda  = valid_borda.sum()
            avg_borda = valid_borda.mean() if len(valid_borda) > 0 else float("nan")
            med_borda = valid_borda.median() if len(valid_borda) > 0 else float("nan")
            pct_top_q = _top_quartile_pct(grp, rank_col) if rank_col in grp.columns else float("nan")

            row[f"longevity_adjusted_borda_{label}"] = round(la_borda, 4)
            row[f"avg_borda_per_window_{label}"]     = round(float(avg_borda), 4) if not math.isnan(avg_borda) else float("nan")
            row[f"median_borda_per_window_{label}"]  = round(float(med_borda), 4) if not math.isnan(med_borda) else float("nan")
            row[f"pct_top_quartile_{label}"]         = pct_top_q

        records.append(row)

    import pandas as pd
    agg = pd.DataFrame(records)

    # Add rank columns for each composite and each individual metric
    for label in all_labels:
        la_col   = f"longevity_adjusted_borda_{label}"
        rank_col = f"borda_rank_{label}"
        if la_col in agg.columns:
            agg[rank_col] = agg[la_col].rank(ascending=False, method="min",
                                              na_option="bottom").astype(int)

    # Sort by primary composite (3-metric, no density)
    agg.sort_values("longevity_adjusted_borda_composite_3",
                    ascending=False, inplace=True)
    agg.reset_index(drop=True, inplace=True)

    return agg


# ============================================================================
# RUN BORDA RANKING
# ============================================================================

def run(per_window_path: str, output_dir: str, max_windows: int) -> None:
    """
    Reads the stacked per-window `ego_metrics` CSV produced by `ego_network.py` and
    computes a longevity-adjusted Borda ranking for each of the four ego-network
    metrics (effective_size, density, constraint, hierarchy) individually. 
    
    Additionally, it computes a composite ranking for different killer profiles.
        - 3-Metric Profile -> (effective_size, constraint, hierarchy)
        - 4-Metric Profile -> 3-Metric + density

    For each Borda variant the aggregated output includes:
        - `longevity_adjusted_borda`  : sum of per-window (k-r)/(k-1) contributions
        - `avg_borda_per_window`      : `longevity_adjusted_borda / windows_present`
                                        (fully comparable across characters regardless
                                        of survival length)
        - `median_borda_per_window`   : median of per-window `(k-r)/(k-1)` scores
                                        (robust to outlier windows)
        - `pct_windows_top_quartile`  : percentage of windows where `rank <= ceil(k*0.25)`
        - `borda_rank`                : rank among all characters on that variant

    All Borda computations use longevity adjustment: each window's contribution
    is `(k - r) / (k - 1)` in [0, 1], where `k = n_alive` and `r = within-window` rank.
    This removes the advantage of protagonists that survive longer and have, thus, a natural
    advantage at ranking well.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ----------------- LOAD DATA -----------------
    print("Loading per-window data...")
    df = stack_window_csvs(Path(per_window_path))
    print(f"  {len(df)} rows, {df['window_lo'].nunique()} windows, "
          f"{df['canonical_id'].nunique()} characters")

    # ----------------- FILTER TO FIRST N WINDOWS -----------------
    if max_windows is not None:
        sorted_los = sorted(df["window_lo"].dropna().unique())
        keep_los   = set(sorted_los[:max_windows])
        df = df[df["window_lo"].isin(keep_los)].copy()

    # ----------------- WITHIN-WINDOW PER-METRIC RANK -----------------
    print("Adding per-metric within-window ranks...")
    df = add_per_metric_ranks(df)

    # ----------------- WITHIN-WINDOW LONG.-ADJ. RANK -----------------
    print("Computing per-window longevity-adjusted Borda contributions...")
    df = add_borda_contributions(df)

    # ----------------- SAVE PER-WINDOW RESULTS -----------------
    per_window_out = os.path.join(output_dir, "borda_per_window.csv")
    keep_cols = (
        ["window_lo", "window_hi", "n_alive", "canonical_id", "name",
         "effective_size", "density", "constraint", "hierarchy"]
        + [f"rank_{label}" for _, _, label in METRIC_RANK_DEFS]
        + ["rank_density_moderate",
           "rank_composite_3", "rank_composite_4"]
        + [f"borda_{label}" for _, _, label in METRIC_RANK_DEFS]
        + ["borda_density_moderate",
           "borda_composite_3", "borda_composite_4"]
    )

    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(per_window_out, index=False)
    print(f"  Saved: {per_window_out}")

    # ----------------- CROSS-WINDOW AGGREGATION -----------------
    print("Aggregating across windows...")
    agg = aggregate_borda(df)

    agg_out = os.path.join(output_dir, "borda_aggregated.csv")
    agg.to_csv(agg_out, index=False)
    print(f"  Saved: {agg_out}")

    # ----------------- PRINT RESULTS -----------------
    print("\n── Borda summary (sorted by 3-metric composite) ──")
    display_cols = [
        "name", "windows_present",
        "longevity_adjusted_borda_composite_3",
        "avg_borda_per_window_composite_3",
        "median_borda_per_window_composite_3",
        "longevity_adjusted_borda_composite_4",
        "avg_borda_per_window_composite_4",
        "borda_rank_composite_3",
        "borda_rank_composite_4",
        "longevity_adjusted_borda_effective_size",
        "longevity_adjusted_borda_constraint",
        "longevity_adjusted_borda_hierarchy",
        "longevity_adjusted_borda_density_moderate",
    ]

    display_cols = [c for c in display_cols if c in agg.columns]
    print(agg[display_cols].to_string(index=False))


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Longevity-adjusted Borda scoring for ego-network killer profiling.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--per-window", required=True,
        help="Path to stacked ego_ranks_per_window.csv or directory of per-window CSVs",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="Only aggregate the first N windows (by window_lo order). Default: all."
    )

    args = parser.parse_args()
    
    # Run Borda Ranking
    run(per_window_path=args.per_window, output_dir=args.output_dir, max_windows=args.max_windows)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
