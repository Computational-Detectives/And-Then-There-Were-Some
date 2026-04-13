from __future__ import annotations

import os
import math
import json
import argparse

from pathlib import Path
from typing import NamedTuple, List, Tuple, Dict, Any


# ============================================================================
# DEATH INTERVAL CONSTANTS
# ============================================================================
class DeathInterval(NamedTuple):
    """
    Encodes one narrative interval between two consecutive deaths.

    Fields
    ------
    label         : human-readable label, e.g. 'Chapter 1', '#1 Marston'
    sentence_start: first sentence ID of the interval (inclusive)
    sentence_end  : last  sentence ID of the interval (inclusive)
    token_start   : first token ID of the interval (inclusive)
    token_end     : last  token ID of the interval (inclusive)
    n_alive       : number of protagonists alive throughout this interval
    victim_name   : canonical fullname of the character who dies *at the end*
                    of this interval (absent from the *next* one).
                    None for the final survivor's interval.
    """
    label:          str
    sentence_start: int
    sentence_end:   int
    token_start:    int
    token_end:      int
    n_alive:        int
    victim_name:    str | None


DEATH_INTERVALS: List[DeathInterval] = [
    DeathInterval("#1 Marston",        1,    1484,     0,  17462, 10, "Anthony James Marston"),
    DeathInterval("#2 E.Rogers",    1485,    1841,  17463, 21070,  9, "Ethel Rogers"),
    DeathInterval("#3 Macarthur",   1842,    2904, 21071, 32904,  8, "John Gordon Macarthur"),
    DeathInterval("#4 T.Rogers",    2905,    3583, 32905, 41438,  7, "Thomas Rogers"),
    DeathInterval("#5 Brent",       3584,    3933, 41439, 45357,  6, "Emily Brent"),
    DeathInterval("#6 Wargrave",    3934,    4416, 45358, 50820,  5, "Lawrence John Wargrave"),
    DeathInterval("#7 Blore",       4417,    5096, 50821, 58126,  4, "William Henry Blore"),
    DeathInterval("#8 Armstrong",   5097,    5195, 58127, 59107,  3, "Edward George Armstrong"),
    DeathInterval("#9 Lombard",     5196,    5279, 59108, 59936,  2, "Philip Lombard"),
    DeathInterval("#10 Claythorne", 5280,    5386, 59937, 60964,  1, None),
]


# Convenience mapping: window_lo → victim name for that window.
VICTIM_BY_WINDOW_LO: Dict[int, str | None] = {
    iv.token_start: iv.victim_name for iv in DEATH_INTERVALS
}
# Sentence-ID version for edges format
VICTIM_BY_WINDOW_LO_SENTENCES: Dict[int, str | None] = {
    iv.sentence_start: iv.victim_name for iv in DEATH_INTERVALS
}


def death_cutpoints(fmt: str = "triples") -> List[int]:
    if fmt == "triples":
        return [iv.token_start for iv in DEATH_INTERVALS]
    return [iv.sentence_start for iv in DEATH_INTERVALS]


def death_schedule_from_intervals(
    intervals: List[DeathInterval],
    fmt: str = "triples",
) -> Dict[str, int]:
    schedule: Dict[str, int] = {}
    for interval in intervals:
        if interval.victim_name is not None:
            last = interval.token_end if fmt == "triples" else interval.sentence_end
            schedule[interval.victim_name] = last
    return schedule


# ============================================================================
# LOADING
# ============================================================================

def _detect_format(path: Path) -> str:
    if "edge_list" in path.stem:
        return "edges"
    return "triples"


def filter_triples_temporal(df, lo: int, hi: int):
    return df[(df["index"] >= lo) & (df["index"] <= hi)].copy()


def load_victims(base_data_dir: str) -> Dict[int, str]:
    """
    Return {canonical_id: fullname} for every main-protagonist character.

    Joins names_owen_split.csv (prot_status == 'main') with
    canonical_mappings.csv on the fullname column.
    """
    import pandas as pd
    names = pd.read_csv(os.path.join(base_data_dir, "names_owen_split.csv"))
    # canonical = pd.read_csv(os.path.join(base_out_dir, "canonical_mappings.csv"))

    # main_names = set(names.loc[names["prot_status"] == "main", "fullname"])
    # victims = canonical[canonical["fullname"].isin(main_names)]
    victims: pd.DataFrame = names.loc[names["prot_status"] == "main", ["id", "fullname"]]
    victims.rename(columns={"id": "canonical_id"}, inplace=True)

    return dict(zip(victims["canonical_id"], victims["fullname"]))


def load_graph_from_triples(path: Path, include_new_triples: bool = False) -> nx.DiGraph:
    """Build a directed graph from AVP or SVO triples TSV."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t")

    if include_new_triples and 'is_new' in df.columns.to_list():
        df = df[~df['is_new']]

    # Normalise column names (SVO uses canonical_subj_id / canonical_obj_id)
    # col_map = {
    #     "canonical_subj_id": "canonical_id_left",
    #     "canonical_obj_id": "canonical_id_right",
    #     "subject_text": "name_left",
    #     "object_text": "name_right",
    #     "verb_id": "index",
    # }
    # df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Remove self-loops
    df = df[df["canonical_id_left"] != df["canonical_id_right"]].copy()

    return df


# ============================================================================
# HELPERS
# ============================================================================

def _get_node_names(G) -> Dict[Any]:
    """Collect {node_id: name} from graph node attributes."""
    return {n: d.get("name", str(n)) for n, d in G.nodes(data=True)}


def _n_alive_for_window(lo: int, hi: int, fmt: str = "triples") -> int:
    """
    Look up the n_alive count for a window [lo, hi] from DEATH_INTERVALS.

    Returns 0 if no interval matches exactly, signalling the caller to
    fall back to counting living victims directly.
    """
    for interval in DEATH_INTERVALS:
        if fmt == "triples":
            if interval.token_start == lo and interval.token_end == hi:
                return interval.n_alive
        else:
            if interval.sentence_start == lo and interval.sentence_end == hi:
                return interval.n_alive
    return 0


def living_victims_in_window(
    victims: Dict[int, str],
    death_schedule: Dict[str, int],
    lo: int,
    hi: int,
) -> Dict[int, str]:
    living = {}
    for vid, name in victims.items():
        last_alive = death_schedule.get(name)
        if last_alive is None or last_alive >= lo:
            living[vid] = name
    return living


# ============================================================================
# GRAPH CREATION
# ============================================================================

# =========== GRAPH CREATION HELPERS ===========
def to_undirected_symmetric(G: nx.DiGraph) -> nx.Graph:
    """
    Convert a DiGraph to an undirected Graph by summing reciprocal edge
    weights explicitly.
    """
    import networkx as nx
    U = nx.Graph()
    U.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        # Skip if we already processed this undirected pair (from the v->u direction)
        if U.has_edge(u, v):
            continue
        w_uv = data.get("weight", 1.0)
        w_vu = G[v][u].get("weight", 0.0) if G.has_edge(v, u) else 0.0
        U.add_edge(u, v, weight=w_uv + w_vu)
    return U


def _alter_only_subgraph(ego_u: nx.Graph, ego_id: int) -> nx.Graph:
    """
    Return the ego-subgraph with ego removed, leaving only alter-alter ties.

    Used for density (Burt defines density over alter-alter ties only) and
    for any calculation that must exclude ego from proportional weights.
    """
    sub = ego_u.copy()
    sub.remove_node(ego_id)
    return sub


# =========== GRAPH CREATION MAIN ===========
def build_nx_graph_triples(df) -> nx.DiGraph:
    import networkx as nx
    agg = (
        df.groupby(["canonical_id_left", "canonical_id_right"])
        .agg(
            weight=("lemma", "size"),
            name_left=("name_left", "first"),
            name_right=("name_right", "first"),
        )
        .reset_index()
    )

    G = nx.from_pandas_edgelist(
        agg,
        source="canonical_id_left",
        target="canonical_id_right",
        edge_attr=["weight"],
        create_using=nx.DiGraph(),
    )

    for _, row in agg.iterrows():
        G.nodes[row["canonical_id_left"]].setdefault("name", row["name_left"])
        G.nodes[row["canonical_id_right"]].setdefault("name", row["name_right"])
    return G


def extract_ego_networks(
    G, victims: Dict[int, str]
) -> Dict[int, nx.Graph]:
    import networkx as nx
    egos = {}
    for vid in victims:
        if vid in G:
            egos[vid] = nx.ego_graph(G, vid, radius=1)
    return egos


# ============================================================================
# ALTER OVERLAP COMPUTATION
# ============================================================================

# =========== ALTER OVERLAP HELPERS ===========

def compute_alter_sets(egos):
    return {vid: set(ego.nodes()) - {vid} for vid, ego in egos.items()}


# =========== ALTER OVERLAP MAIN ===========

def pairwise_overlap_matrix(alter_sets, victims):
    import pandas as pd
    ids = sorted(alter_sets.keys())
    labels = [victims.get(v, str(v)) for v in ids]
    mat = pd.DataFrame(0, index=labels, columns=labels)
    for i, vi in enumerate(ids):
        for j, vj in enumerate(ids):
            mat.iloc[i, j] = len(alter_sets[vi] & alter_sets[vj])
    return mat


def compute_k_overlap(alter_sets, victims, node_names, min_k=2):
    from collections import Counter
    import pandas as pd
    counter: Counter = Counter()
    alter_to_victims: Dict[Any] = {}
    for vid, alters in alter_sets.items():
        for a in alters:
            counter[a] += 1
            alter_to_victims.setdefault(a, []).append(victims.get(vid, str(vid)))
    rows = []
    for alter_id, count in counter.most_common():
        if count < min_k:
            continue
        rows.append({
            "canonical_id":        alter_id,
            "name":                node_names.get(alter_id, str(alter_id)),
            "victim_overlap_count": count,
            "shared_with_victims": "; ".join(alter_to_victims[alter_id]),
        })
    return pd.DataFrame(rows)


# ============================================================================
# EGO NETWORK METRICS
# ============================================================================

# =========== EGO NETWORK METRICS HELPERS ===========

def _dyadic_constraint(ego_u: nx.Graph, ego_id: int) -> Dict[int, float]:
    """
    Compute Burt's dyadic constraint `c_ij` for each alter `j` of ego `i`.

        ``c_ij = (p_ij + Σ_q p_iq * p_qj)²``

    where `p_ij = w_ij / Σ_k w_ik`  (proportional tie strength, ego-relative)
    and   `p_qj = w_qj / Σ_r w_qr`  (proportional tie strength, alter-relative,
                                     computed excluding ego from `q`'s neighbours)

    Essentially equivalent to ``networkx.local_constraint``, but excludes ego 
    when computing alter-alter tie strengths i.e., indirect path of constraint.

    It also collects the distribution over all ego-alter constraint values.

    :return: ``{alter_id: c_ij}`` for every neighbour `j` of `ego_id`.
    """
    neighbours = [n for n in ego_u.neighbors(ego_id)]
    if not neighbours:
        return {}

    # --- p_ij: ego's proportional tie strengths to each alter ---
    raw_weights = {j: ego_u[ego_id][j].get("weight", 1.0) for j in neighbours}
    total_w = sum(raw_weights.values())
    p_ego = {j: w / total_w for j, w in raw_weights.items()} if total_w > 0 else {}

    # --- Precompute p_qj for every (q, j) pair: q's proportion to j,
    #     excluding ego from q's neighbour list ---
    # p_alter[q][j] = proportion of q's tie strength going to j (ego excluded)
    p_alter: Dict[int, Dict[int, float]] = {}
    for q in neighbours:
        q_neighbours = [r for r in ego_u.neighbors(q) if r != ego_id]  # FIX 1
        total_q = sum(ego_u[q][r].get("weight", 1.0) for r in q_neighbours)
        if total_q > 0:
            p_alter[q] = {
                r: ego_u[q][r].get("weight", 1.0) / total_q
                for r in q_neighbours
            }
        else:
            p_alter[q] = {}

    # --- Dyadic constraint ---
    dyadic: Dict[int, float] = {}
    for j in neighbours:
        direct = p_ego.get(j, 0.0)

        # Indirect: sum over mutual alters q (q ≠ j)
        indirect = 0.0
        for q in neighbours:
            if q == j:
                continue
            p_iq = p_ego.get(q, 0.0)
            p_qj = p_alter.get(q, {}).get(j, 0.0)
            indirect += p_iq * p_qj

        dyadic[j] = (direct + indirect) ** 2

    return dyadic


def _hierarchy(dyadic: Dict[int, float]) -> float:
    """
    Burt's hierarchy: concentration of constraint across alters.

        `H = (Σ_j s_j * ln(s_j)) / (N * ln(1/N))`

    where `s_j = c_ij / C_i` is alter `j`'s share of total constraint.

    Returns a value in [0, 1]:
        1.0 → all constraint from a single alter (or n == 1)
        0.0 → constraint evenly spread across all alters
        NaN → no positive constraint values (undefined)

    NOTE: zero-valued c_ij are excluded before computing n and shares,
    consistent with Burt's treatment.  This means n here may be smaller
    than ego_size; this is intentional and documented.
    """
    vals = [v for v in dyadic.values() if v > 0]
    n = len(vals)

    if n == 0:
        return float("nan")

    if n == 1:
        # Single constraining alter → hierarchy is maximally concentrated
        return 1.0

    total_c = sum(vals)
    if total_c == 0:
        return float("nan")

    shares = [v / total_c for v in vals]

    log_n = math.log(n)
    if log_n == 0:
        return float("nan")

    entropy_sum = sum(s * math.log(s) for s in shares if s > 0)
    return 1.0 + entropy_sum / log_n


# =========== EGO NETWORK METRICS MAIN ===========

def compute_ego_metrics(
    egos: Dict[int, "nx.Graph | nx.DiGraph"],
    victims: Dict[int, str],
    window_lo: int | None = None,
    window_hi: int | None = None,
    n_alive: int | None = None,
) -> pd.DataFrame:
    """
    Compute structural metrics for each ego-network.

    Args
    ----
    G_full              : full window interaction graph (all nodes, not just
                          ego-subgraphs).  Required for NEW 5 and NEW 6.
                          If None those columns will be NaN.
    victim_by_window_lo : {window_lo: victim_name} mapping used to look up
                          the upcoming victim for Jaccard (NEW 4).
                          If None, Jaccard columns will be NaN.
    """
    import networkx as nx
    import pandas as pd

    rows = []

    for vid, name in victims.items():
        # Absent ego
        if vid not in egos:
            rows.append({
                "window_lo":                    window_lo,
                "window_hi":                    window_hi,
                "n_alive":                      n_alive,
                "canonical_id":                 vid,
                "name":                         name,
                "ego_size":                     0,
                "density":                      None,
                "effective_size":               None,
                "constraint":                   None,
                "hierarchy":                    None,
                "degree":                       0,
                # Full dyadic distribution
                "all_dyadic_constraint":        None,
                # Most constraining alter
                "most_constraining_alter":      None,
                "alter_dyadic_constraint":      None,
            })
            continue

        ego = egos[vid]

        if ego.is_directed():
            # Undirected edges through summation of reciprocal edge weights
            ego_u = to_undirected_symmetric(ego)
        else:
            ego_u = ego

        n_total = ego_u.number_of_nodes()
        size = n_total - 1  # alters only

        # Alter-only subgraph to comply with Burt's density definition
        alter_sub = _alter_only_subgraph(ego_u, vid)
        n_alters = alter_sub.number_of_nodes()
        n_alter_edges = alter_sub.number_of_edges()

        density = nx.density(alter_sub) if n_alters > 1 else 0.0

        # Compute ego-measures on whole ego-graph
        try:
            eff = nx.effective_size(ego_u).get(vid, float("nan"))
        except Exception:
            eff = float("nan")

        try:
            con = nx.constraint(ego_u).get(vid, float("nan"))
        except Exception:
            con = float("nan")

        degree = ego_u.degree(vid) if vid in ego_u else 0

        # Calculate dyadic constraint for ego-graph
        dyadic = _dyadic_constraint(ego_u, vid)

        # Serialise full distribution as JSON {alter_name: c_ij}
        alter_names = {n: ego_u.nodes[n].get("name", str(n)) for n in ego_u.nodes}
        all_dyadic_json = json.dumps(
            {alter_names.get(j, str(j)): round(v, 6) for j, v in dyadic.items()},
            ensure_ascii=False,
        ) if dyadic else None

        # Determine most constraining alter
        if dyadic:
            top_j = max(dyadic, key=dyadic.get)
            top_name = alter_names.get(top_j, str(top_j))
            top_c = dyadic[top_j]
        else:
            top_name = None
            top_c = float("nan")

        # Calculate hierarchy for ego-graph
        hier = _hierarchy(dyadic)

        def _fmt(v):
            return round(v, 4) if (v is not None and not math.isnan(v)) else None
        
        rows.append({
            # ── Window metadata ────────────────────────────────────────────
            "window_lo":                    window_lo,
            "window_hi":                    window_hi,
            "n_alive":                      n_alive,
            # ── Identity ──────────────────────────────────────────────────
            "canonical_id":                 vid,
            "name":                         victims.get(vid, str(vid)),
            # ── Raw structural metrics ─────────────────────────────────────
            "ego_size":                     size,
            "density":                      _fmt(density),
            "effective_size":               _fmt(eff),
            "constraint":                   _fmt(con),
            "hierarchy":                    _fmt(hier),
            "degree":                       degree,
            # ── Full dyadic constraint distribution ────────────────────────
            "all_dyadic_constraint":        all_dyadic_json,
            # ── Most constraining alter ────────────────────────────────────
            "most_constraining_alter":      top_name,
            "alter_dyadic_constraint":      _fmt(top_c),
        })

    return pd.DataFrame(rows)


# ============================================================================
# EGO NETWORK ANALYSIS
# ============================================================================

def run_analysis(
    df_raw: pd.DataFrame,
    fmt: str,
    victims: Dict[int, str],
    windows: List[Tuple[int, int]],
    min_k: int,
    out_root: Path,
    web: bool,
) -> None:
    """
    Main analysis loop.  Iterates over temporal windows, builds graphs,
    extracts ego-networks, computes metrics and overlaps, and writes outputs.

    Only those victims that are alive at the start of each window are passed to
    ``extract_ego_networks()``, so the dead characters never appear as egos
    in windows after their death.

    ``window_lo``, ``window_hi``, and ``n_alive`` are stamped onto every row of
    ``ego_metrics.csv`` so files are self-describing and stackable.
    """
    death_schedule = death_schedule_from_intervals(DEATH_INTERVALS, fmt=fmt)

    for lo, hi in windows:
        out_dir = out_root if len(windows) == 1 else out_root / f"window_{lo}_{hi}"
        os.makedirs(out_dir, exist_ok=True)

        # ------------------ DETERMINE #ALIVE EGO -------------------
        n_alive = _n_alive_for_window(lo, hi, fmt=fmt)
        if n_alive == 0:
            living_all = living_victims_in_window(victims, death_schedule, lo, hi)
            n_alive = len(living_all)

        # ------------------ BUILD FULL GRAPH FOR GIVEN WINDOWS -------------------
        if fmt == "triples":
            df = filter_triples_temporal(df_raw, lo, hi)
            G = build_nx_graph_triples(df)

        if G.number_of_nodes() == 0:
            print(f"  [window {lo}–{hi}] Graph is empty — skipping.")
            continue

        # ------------------ LOAD LIVING VICTIMS -------------------
        living = living_victims_in_window(victims, death_schedule, lo, hi)
        if not living:
            print(f"  [window {lo}–{hi}] No living victims — skipping.")
            continue
        
        # ------------------ COMPUTE EGO GRAPH -------------------
        egos = extract_ego_networks(G, living)
        if not egos:
            print(f"  [window {lo}–{hi}] No living victims found in graph — skipping.")
            continue

        # ------------------ COMPUTE EGO METRICS -------------------
        metrics_df = compute_ego_metrics(
            egos, living,
            window_lo=lo,
            window_hi=hi,
            n_alive=n_alive,
        )
        metrics_df.to_csv(out_dir / "ego_metrics.csv", index=False)
        print(f"  [window {lo}–{hi}] ego_metrics.csv  ({len(metrics_df)} victims, n_alive={n_alive})")

        # ------------------ COMPUTE ALTER OVERLAP -------------------
        alter_sets = compute_alter_sets(egos)
        node_names = _get_node_names(G)

        overlap_mat = pairwise_overlap_matrix(alter_sets, living)
        overlap_mat.to_csv(out_dir / "pairwise_overlap.csv")

        k_df = compute_k_overlap(alter_sets, living, node_names, min_k)
        k_df.to_csv(out_dir / "alter_overlap.csv", index=False)
        print(f"  [window {lo}–{hi}] alter_overlap.csv ({len(k_df)} alters with overlap ≥ {min_k})")

        # ------------------ DETERMINE COMMON EGOS IN ALTER SETS -------------------
        if alter_sets:
            full_intersection = set.intersection(*alter_sets.values())
            if full_intersection:
                inter_names = [node_names.get(a, str(a)) for a in full_intersection]
                print(f"  [window {lo}–{hi}] Full intersection across all victims: {inter_names}")
            else:
                print(f"  [window {lo}–{hi}] No character appears in ALL victim ego-networks.")

        # ------------------ VISUALISE GRAPHS -------------------
        if web:
            from .visualization.ego_network_viz import (
                visualise_ego, visualise_shared_alters, visualise_heatmap,
            )
            for vid, ego in egos.items():
                visualise_ego(ego, vid, living[vid], out_dir)
            shared_ids = set(k_df["canonical_id"]) if not k_df.empty else set()
            visualise_shared_alters(G, shared_ids, set(living.keys()), out_dir)
            visualise_heatmap(overlap_mat, out_dir)
            print(f"  [window {lo}–{hi}] Visualisations saved.")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    from ..config import DATA_DIR, OBJ_OUT, EGO_OUT
    from ..auxiliary import int_list, make_windows

    description = (
        "Ego-network analysis for perpetrator detection.\n\n"
        "Builds ego-networks for each victim, computes structural metrics,\n"
        "determines alter overlap, and produces interactive visualisations."
    )

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=Path,
        default=Path(OBJ_OUT) / "triples.csv",
        help="Path to the input file (triples TSV or edge-list CSV)",
    )
    parser.add_argument(
        "-f", "--format", choices=["triples", "edges"], default=None,
        help="Input format. Auto-detected if omitted.",
    )
    parser.add_argument(
        "-k", "--min-overlap", type=int, default=2,
        help="Minimum victim count for an alter to be flagged (default: 2)",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=Path(EGO_OUT),
        help="Output directory",
    )
    parser.add_argument(
        "-w", "--web", action="store_true",
        help="Generate interactive pyvis HTML visualisations",
    )
    parser.add_argument(
        "-t", "--cutpoints", type=int_list, default=None,
        help=(
            "Comma-separated cutpoints for temporal windowing, e.g. '5000,10000'.\n"
            "For triples these are token IDs (index column);\n"
            "for edge lists these are sentence IDs.\n"
            "If omitted, a single static analysis over the full text is performed."
        ),
    )

    parser.add_argument(
        "--include-new-triples", action="store_true", default=False,
        help=(
            "Include manually added triples. (Default=False)"
        )
    )

    args = parser.parse_args()

    fmt = args.format or _detect_format(args.input)
    print(f"Input : {args.input}")
    print(f"Format: {fmt}")

    if fmt == "triples":
        df_raw = load_graph_from_triples(args.input, args.include_new_triples)
        temporal_col_min = DEATH_INTERVALS[0].token_start
        temporal_col_max = DEATH_INTERVALS[-1].token_end

    victims = load_victims(DATA_DIR)
    print(f"Victims ({len(victims)}): {list(victims.values())}")

    cutpoints = args.cutpoints if args.cutpoints is not None else death_cutpoints(fmt)
    windows = make_windows(cutpoints, temporal_col_min, temporal_col_max)
    print(f"Windows: {windows}\n")

    run_analysis(
        df_raw, fmt, victims, windows,
        args.min_overlap, args.out, args.web,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
