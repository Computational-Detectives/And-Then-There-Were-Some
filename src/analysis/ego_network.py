# Lazily bind type annotations
from __future__ import annotations

import os
import argparse
from pathlib import Path

from ..config import (
    BASE_DATA_DIR,
    BASE_OUT_DIR,
    TRIPLE_OUT,
    EGO_OUT,
)

from ..auxiliary import int_list, make_windows


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def _detect_format(path: Path) -> str:
    """Heuristic: if the filename contains 'edge_list' treat as edges, else triples."""
    if "edge_list" in path.stem:
        return "edges"
    return "triples"


def load_victims(base_data_dir: str, base_out_dir: str) -> dict[int, str]:
    """
    Return {canonical_id: fullname} for every main-protagonist character.

    Joins names_owen_split.csv (prot_status == 'main') with
    canonical_mappings.csv on the fullname column.
    """
    import pandas as pd
    names = pd.read_csv(os.path.join(base_data_dir, "names_owen_split.csv"))
    canonical = pd.read_csv(os.path.join(base_out_dir, "canonical_mappings.csv"))

    main_names = set(names.loc[names["prot_status"] == "main", "fullname"])
    victims = canonical[canonical["fullname"].isin(main_names)]

    return dict(zip(victims["canonical_id"], victims["fullname"]))


def load_graph_from_triples(path: Path) -> nx.DiGraph:
    """Build a directed graph from AVP or SVO triples TSV."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t")

    # Normalise column names (SVO uses canonical_subj_id / canonical_obj_id)
    col_map = {
        "canonical_subj_id": "canonical_id_left",
        "canonical_obj_id": "canonical_id_right",
        "subject_text": "name_left",
        "object_text": "name_right",
        "verb_id": "index",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Remove self-loops
    df = df[df["canonical_id_left"] != df["canonical_id_right"]].copy()

    return df


def load_graph_from_edges(path: Path) -> pd.DataFrame:
    """Load the co-occurrence edge list CSV."""
    import ast
    import pandas as pd
    df = pd.read_csv(path)

    # Parse sentence_ids from string to list
    if "sentence_ids" in df.columns:
        df["sentence_ids"] = df["sentence_ids"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    return df


# ─────────────────────────────────────────────
# 2. TEMPORAL FILTERING
# ─────────────────────────────────────────────
def filter_triples_temporal(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Keep rows whose token-index falls in [lo, hi]."""
    return df[(df["index"] >= lo) & (df["index"] <= hi)].copy()


def filter_edges_temporal(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Keep edges whose sentence_ids list has at least one ID in [lo, hi]."""
    mask = df["sentence_ids"].apply(
        lambda ids: any(lo <= s <= hi for s in ids)
    )
    return df[mask].copy()


# ─────────────────────────────────────────────
# 3. GRAPH BUILDING
# ─────────────────────────────────────────────
def build_nx_graph_triples(df: pd.DataFrame) -> nx.DiGraph:
    """Build a weighted DiGraph from filtered triples DataFrame."""
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

    # Node labels
    for _, row in agg.iterrows():
        G.nodes[row["canonical_id_left"]].setdefault("name", row["name_left"])
        G.nodes[row["canonical_id_right"]].setdefault("name", row["name_right"])

    return G


def build_nx_graph_edges(df: pd.DataFrame) -> nx.Graph:
    """Build a weighted undirected Graph from the co-occurrence edge list."""
    import networkx as nx
    G = nx.from_pandas_edgelist(
        df,
        source="source_id",
        target="target_id",
        edge_attr=["weight"],
        create_using=nx.Graph(),
    )

    for _, row in df.iterrows():
        G.nodes[row["source_id"]].setdefault("name", row["source_name"])
        G.nodes[row["target_id"]].setdefault("name", row["target_name"])

    return G


# ─────────────────────────────────────────────
# 4. EGO-NETWORK EXTRACTION & METRICS
# ─────────────────────────────────────────────

def extract_ego_networks(
    G: nx.Graph | nx.DiGraph, victims: dict[int, str]
) -> dict[int, nx.Graph | nx.DiGraph]:
    """Return {victim_id: ego_subgraph} for every victim present in G."""
    import networkx as nx
    egos = {}
    for vid in victims:
        if vid in G:
            egos[vid] = nx.ego_graph(G, vid, radius=1)
    return egos


def _dyadic_constraint(G_u: nx.Graph, ego_id: int) -> dict[int, float]:
    """
    Compute Burt's dyadic constraint c_ij for each alter j of ego i.

    c_ij = (p_ij + Σ_q p_iq * p_qj)²

    where p_ij = w_ij / Σ_k w_ik  (proportional tie strength).

    Returns {alter_id: c_ij} for every neighbour j of ego_id.
    """
    neighbours = list(G_u.neighbors(ego_id))
    if not neighbours:
        return {}

    # Proportional tie strengths  p_ij
    weights = {}
    total_w = 0.0
    for j in neighbours:
        w = G_u[ego_id][j].get("weight", 1.0)
        weights[j] = w
        total_w += w

    p = {j: w / total_w for j, w in weights.items()} if total_w > 0 else {}

    # Dyadic constraint
    dyadic = {}
    for j in neighbours:
        # Direct proportion
        direct = p.get(j, 0.0)

        # Indirect paths through mutual alters q (q ≠ i, q ≠ j)
        indirect = 0.0
        for q in neighbours:
            if q == j:
                continue
            p_iq = p.get(q, 0.0)
            # p_qj: proportion of q's ties going to j
            q_neighbours = list(G_u.neighbors(q))
            total_q = sum(G_u[q][r].get("weight", 1.0) for r in q_neighbours)
            if total_q > 0 and G_u.has_edge(q, j):
                p_qj = G_u[q][j].get("weight", 1.0) / total_q
            else:
                p_qj = 0.0
            indirect += p_iq * p_qj

        dyadic[j] = (direct + indirect) ** 2

    return dyadic


def _hierarchy(dyadic: dict[int, float]) -> float:
    """
    Burt's hierarchy: how concentrated the constraint is in one alter.

    H = (Σ_j (c_ij / C_i) * ln(c_ij / C_i)) / (N * ln(1/N))

    Returns a value in [0, 1].  1 = all constraint from one alter,
    0 = constraint evenly spread.  Returns NaN if fewer than 2 alters.
    """
    import math

    vals = [v for v in dyadic.values() if v > 0]
    n = len(vals)
    if n < 2:
        return float("nan")

    total_c = sum(vals)
    if total_c == 0:
        return float("nan")

    shares = [v / total_c for v in vals]

    numerator = sum(s * math.log(s) for s in shares if s > 0)
    denominator = n * math.log(1.0 / n)

    if denominator == 0:
        return float("nan")

    return numerator / denominator


def compute_ego_metrics(
    egos: dict[int, nx.Graph | nx.DiGraph],
    victims: dict[int, str],
) -> pd.DataFrame:
    """Compute structural metrics for each ego-network.

    Metrics include size, density, effective size, constraint, hierarchy
    (Burt), degree, and identification of the most constraining alter.
    """
    import networkx as nx
    
    rows = []
    for vid, ego in egos.items():
        n = ego.number_of_nodes()
        size = n - 1  # exclude ego

        # For metrics that need an undirected view
        ego_u = ego.to_undirected() if ego.is_directed() else ego

        density = nx.density(ego_u) if n > 1 else 0.0

        # Effective size and constraint (Burt) – networkx computes on undirected
        try:
            eff_size = nx.effective_size(ego_u)
            eff = eff_size.get(vid, float("nan"))
        except Exception:
            eff = float("nan")

        try:
            constraint = nx.constraint(ego_u)
            con = constraint.get(vid, float("nan"))
        except Exception:
            con = float("nan")

        degree = ego.degree(vid) if vid in ego else 0

        # --- Hierarchy & most constraining alter ---
        dyadic = _dyadic_constraint(ego_u, vid)
        hier = _hierarchy(dyadic)

        if dyadic:
            top_alter_id = max(dyadic, key=dyadic.get)
            top_alter_name = ego_u.nodes[top_alter_id].get("name", str(top_alter_id))
            top_alter_c = dyadic[top_alter_id]
        else:
            top_alter_name = None
            top_alter_c = float("nan")

        rows.append(
            {
                "canonical_id": vid,
                "name": victims.get(vid, str(vid)),
                "ego_size": size,
                "density": round(density, 4),
                "effective_size": round(eff, 4) if eff == eff else None,
                "constraint": round(con, 4) if con == con else None,
                "hierarchy": round(hier, 4) if hier == hier else None,
                "most_constraining_alter": top_alter_name,
                "alter_dyadic_constraint": round(top_alter_c, 4) if top_alter_c == top_alter_c else None,
                "degree": degree,
            }
        )

    import pandas as pd
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 5. ALTER OVERLAP ANALYSIS
# ─────────────────────────────────────────────

def compute_alter_sets(
    egos: dict[int, nx.Graph | nx.DiGraph],
) -> dict[int, set[int]]:
    """Return {victim_id: set_of_alter_ids} (excluding the ego itself)."""
    return {vid: set(ego.nodes()) - {vid} for vid, ego in egos.items()}


def pairwise_overlap_matrix(
    alter_sets: dict[int, set[int]], victims: dict[int, str]
) -> pd.DataFrame:
    """Build a |V|×|V| matrix of shared-alter counts."""
    
    ids = sorted(alter_sets.keys())
    labels = [victims.get(v, str(v)) for v in ids]
    mat = pd.DataFrame(0, index=labels, columns=labels)

    for i, vi in enumerate(ids):
        for j, vj in enumerate(ids):
            mat.iloc[i, j] = len(alter_sets[vi] & alter_sets[vj])

    return mat


def compute_k_overlap(
    alter_sets: dict[int, set[int]],
    victims: dict[int, str],
    node_names: dict[int, str],
    min_k: int = 2,
) -> pd.DataFrame:
    """
    For every alter, count how many victim ego-networks it appears in.
    Return only those appearing in >= min_k victims.
    """
    from collections import Counter

    counter: Counter[int] = Counter()
    alter_to_victims: dict[int, list[str]] = {}

    for vid, alters in alter_sets.items():
        for a in alters:
            counter[a] += 1
            alter_to_victims.setdefault(a, []).append(victims.get(vid, str(vid)))

    rows = []
    for alter_id, count in counter.most_common():
        if count < min_k:
            continue
        rows.append(
            {
                "canonical_id": alter_id,
                "name": node_names.get(alter_id, str(alter_id)),
                "victim_overlap_count": count,
                "shared_with_victims": "; ".join(alter_to_victims[alter_id]),
            }
        )
    
    import pandas as pd
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 7. ORCHESTRATION
# ─────────────────────────────────────────────

def _get_node_names(G: nx.Graph | nx.DiGraph) -> dict[int, str]:
    """Collect {node_id: name} from graph node attributes."""
    return {n: d.get("name", str(n)) for n, d in G.nodes(data=True)}


def run_analysis(
    df_raw: pd.DataFrame,
    fmt: str,
    victims: dict[int, str],
    windows: list[tuple[int, int]],
    min_k: int,
    out_root: Path,
    web: bool,
) -> None:
    """
    Main analysis loop.  Iterates over temporal windows, builds graphs,
    extracts ego-networks, computes metrics and overlaps, and writes outputs.
    """
    for lo, hi in windows:
        # --- Determine output directory ---
        if len(windows) == 1:
            out_dir = out_root
        else:
            out_dir = out_root / f"window_{lo}_{hi}"

        os.makedirs(out_dir, exist_ok=True)

        # --- Temporal filter ---
        if fmt == "triples":
            df = filter_triples_temporal(df_raw, lo, hi)
            G = build_nx_graph_triples(df)
        else:
            df = filter_edges_temporal(df_raw, lo, hi)
            G = build_nx_graph_edges(df)

        if G.number_of_nodes() == 0:
            print(f"  [window {lo}–{hi}] Graph is empty — skipping.")
            continue

        # --- Ego-networks ---
        egos = extract_ego_networks(G, victims)
        if not egos:
            print(f"  [window {lo}–{hi}] No victims found in graph — skipping.")
            continue

        # --- Metrics ---
        metrics_df = compute_ego_metrics(egos, victims)
        metrics_df.to_csv(out_dir / "ego_metrics.csv", index=False)
        print(f"  [window {lo}–{hi}] ego_metrics.csv  ({len(metrics_df)} victims)")

        # --- Overlap ---
        alter_sets = compute_alter_sets(egos)
        node_names = _get_node_names(G)

        overlap_mat = pairwise_overlap_matrix(alter_sets, victims)
        overlap_mat.to_csv(out_dir / "pairwise_overlap.csv")

        k_df = compute_k_overlap(alter_sets, victims, node_names, min_k)
        k_df.to_csv(out_dir / "alter_overlap.csv", index=False)
        print(f"  [window {lo}–{hi}] alter_overlap.csv ({len(k_df)} alters with overlap ≥ {min_k})")

        # --- Full intersection ---
        if alter_sets:
            full_intersection = set.intersection(*alter_sets.values())
            if full_intersection:
                inter_names = [node_names.get(a, str(a)) for a in full_intersection]
                print(f"  [window {lo}–{hi}] Full intersection across all victims: {inter_names}")
            else:
                print(f"  [window {lo}–{hi}] No character appears in ALL victim ego-networks.")

        # --- Visualisations ---
        if web:
            from .visualization.ego_network_viz import (
                visualise_ego, visualise_shared_alters, visualise_heatmap,
            )
            # Per-victim ego HTML
            for vid, ego in egos.items():
                visualise_ego(ego, vid, victims[vid], out_dir)

            # Shared-alter subgraph
            shared_ids = set(k_df["canonical_id"]) if not k_df.empty else set()
            visualise_shared_alters(G, shared_ids, set(victims.keys()), out_dir)

            # Heatmap
            visualise_heatmap(overlap_mat, out_dir)
            print(f"  [window {lo}–{hi}] Visualisations saved.")


# ─────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────

def main() -> None:
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
        default=Path(f"{TRIPLE_OUT}/avp_triples.csv"),
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

    args = parser.parse_args()

    # --- Detect format ---
    fmt = args.format or _detect_format(args.input)
    print(f"Input : {args.input}")
    print(f"Format: {fmt}")

    # --- Load data ---
    if fmt == "triples":
        df_raw = load_graph_from_triples(args.input)
        temporal_col_min = int(df_raw["index"].min())
        temporal_col_max = int(df_raw["index"].max())
    else:
        df_raw = load_graph_from_edges(args.input)
        # Flatten sentence_ids to get range
        all_sids = [s for ids in df_raw["sentence_ids"] for s in ids]
        temporal_col_min = min(all_sids)
        temporal_col_max = max(all_sids)

    # --- Load victims ---
    victims = load_victims(BASE_DATA_DIR, BASE_OUT_DIR)
    print(victims)
    return
    print(f"Victims ({len(victims)}): {list(victims.values())}")

    # --- Build windows ---
    windows = make_windows(args.cutpoints, temporal_col_min, temporal_col_max)
    print(f"Windows: {windows}\n")

    # --- Run ---
    run_analysis(df_raw, fmt, victims, windows, args.min_overlap, args.out, args.web)
    print("\nDone.")


if __name__ == "__main__":
    main()
