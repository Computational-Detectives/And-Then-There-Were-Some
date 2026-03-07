"""
Character timeline — orchestration and CLI.

Data loading / binning lives in timeline_data.py.
Plotting helpers live in visualisation/timeline_viz.py.
"""
from __future__ import annotations

from pathlib import Path

from ..config import (
    BASE_DATA_DIR,
    BASE_OUT_DIR,
    COOC_OUT,
    EGO_OUT,
    TOKENS,
)

from ..auxiliary import (
    short_name,
    int_pair,
    build_chapter_sentence_ranges,
    format_chapter_ranges,
    token_range_to_sentence_range,
)

from .timeline_data import (
    load_victims,
    build_mention_series,
    _bin_equal,
    _bin_normalised,
    _bin_chapters,
    _build_matrix,
    _detect_hierarchy_format,
    _load_and_normalise_triples,
    _build_graph_from_cooc_slice,
    _build_graph_from_triples_slice,
)

from .visualization.timeline_viz import (
    plot_heatmap,
    plot_streamgraph,
    plot_lines,
    plot_lines_with_derivative,
    plot_hierarchy_timeline,
)


# ─────────────────────────────────────────────
# HIERARCHY COMPUTATION
# ─────────────────────────────────────────────

def _compute_hierarchy_matrix(
    data: pd.DataFrame,
    graph_builder,
    victims: dict[int, str],
    n_bins: int,
    tokens_path: str,
    bin_mode: str,
    sid_range: tuple[int, int] | None,
) -> tuple[np.ndarray, list[str], np.ndarray | None, np.ndarray, list[str], str]:
    """
    Core hierarchy computation shared by both cooccurrence and triples inputs.

    Parameters
    ----------
    data : DataFrame with a ``sentence_id`` column and the columns needed by
           *graph_builder*.
    graph_builder : callable(df_slice) → nx.Graph
    """
    from .ego_network import _dyadic_constraint, _hierarchy, extract_ego_networks

    # Apply sentence range filter
    if sid_range is not None:
        sid_lo, sid_hi = sid_range
        data = data[
            (data["sentence_id"] >= sid_lo) & (data["sentence_id"] <= sid_hi)
        ].copy()

    # --- determine bin edges ---
    if bin_mode == "chapters":
        chapters = build_chapter_sentence_ranges(tokens_path)
        if sid_range is not None:
            chapters = [
                ch for ch in chapters
                if ch["end_sid"] >= sid_range[0] and ch["start_sid"] <= sid_range[1]
            ]
        bin_edges = [(ch["start_sid"], ch["end_sid"]) for ch in chapters]
        tick_pos = np.arange(len(bin_edges))
        tick_lab = [f"Ch {ch['chapter']}" for ch in chapters]
        edges_arr = None
        xlabel = "Chapter"
    else:
        min_s = int(data["sentence_id"].min())
        max_s = int(data["sentence_id"].max())
        raw_edges = np.linspace(min_s, max_s + 1, n_bins + 1)
        bin_edges = [(int(raw_edges[i]), int(raw_edges[i + 1]) - 1) for i in range(n_bins)]
        n_ticks = min(n_bins, 20)
        tick_pos = np.linspace(0, len(bin_edges) - 1, n_ticks, dtype=int)
        tick_lab = [f"{int(raw_edges[t])}" for t in tick_pos]
        edges_arr = raw_edges
        xlabel = "Sentence ID (narrative progression →)"

    # --- compute hierarchy per bin per character ---
    char_ids = sorted(victims.keys())
    n_chars = len(char_ids)
    n_actual_bins = len(bin_edges)
    hier_matrix = np.full((n_chars, n_actual_bins), np.nan)

    for b, (lo, hi) in enumerate(bin_edges):
        slc = data[(data["sentence_id"] >= lo) & (data["sentence_id"] <= hi)]
        if slc.empty:
            continue

        G = graph_builder(slc)
        if G.number_of_nodes() == 0:
            continue
        egos = extract_ego_networks(G, victims)

        for vid, ego in egos.items():
            ego_u = ego.to_undirected() if ego.is_directed() else ego
            dyadic = _dyadic_constraint(ego_u, vid)
            h = _hierarchy(dyadic)
            idx = char_ids.index(vid)
            hier_matrix[idx, b] = h

    # Sort by average hierarchy (descending; NaN-safe)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg = np.nanmean(hier_matrix, axis=1)
    order = np.argsort(-np.nan_to_num(avg, nan=-1))
    hier_sorted = hier_matrix[order]
    labels = [short_name(victims[char_ids[o]]) for o in order]

    return hier_sorted, labels, edges_arr, tick_pos, tick_lab, xlabel


# ─────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────

def run_mode(
    mentions: pd.DataFrame,
    victims: dict[int, str],
    bin_mode: str,
    n_bins: int,
    tokens_path: str,
    out_dir: Path,
    kind: str,
) -> None:
    # ── Binning ──
    if bin_mode == "chapters":
        mentions_b, nb, tp, tl, edges, xlabel = _bin_chapters(mentions, tokens_path)
        suffix = "_chapters"
    elif bin_mode == "normalised":
        mentions_b, nb, tp, tl, edges, xlabel = _bin_normalised(mentions, n_bins)
        suffix = "_normalised"
    else:
        mentions_b, nb, tp, tl, edges, xlabel = _bin_equal(mentions, n_bins)
        suffix = ""

    char_ids = sorted(victims.keys())
    matrix = _build_matrix(mentions_b, char_ids, nb)

    # Sentence-ID range for death shading
    sid_min = int(mentions["sentence_id"].min())
    sid_max = int(mentions["sentence_id"].max())
    srange = (sid_min, sid_max)

    # Sort rows by total mentions (descending)
    totals = matrix.sum(axis=1)
    order = np.argsort(-totals)
    matrix_sorted = matrix[order]
    labels = [short_name(victims[char_ids[o]]) for o in order]

    print(f"\n[{bin_mode}] {nb} bins")
    plot_heatmap(matrix_sorted, labels, tp, tl, xlabel, out_dir, suffix,
                 sid_range=srange)
    plot_streamgraph(matrix_sorted, labels, nb, edges, xlabel, out_dir, suffix,
                     tick_pos=tp, tick_lab=tl, sid_range=srange)
    plot_lines(matrix_sorted, labels, nb, edges, xlabel, out_dir, suffix,
               tick_pos=tp, tick_lab=tl, sid_range=srange)
    plot_lines_with_derivative(matrix_sorted, labels, nb, edges, xlabel, out_dir, suffix,
                               tick_pos=tp, tick_lab=tl, kind=kind, sid_range=srange)


def run_hierarchy(
    hier_input: Path,
    victims: dict[int, str],
    n_bins: int,
    tokens_path: str,
    out_dir: Path,
    modes: list[str],
    sid_range: tuple[int, int] | None = None,
) -> None:
    """Compute and plot hierarchy over time for each requested bin mode.

    Accepts either a raw_cooccurrences CSV or a triples TSV (AVP/SVO).
    Format is auto-detected from the file header.
    """
    fmt = _detect_hierarchy_format(hier_input)
    print(f"\n  Hierarchy input: {hier_input}  (format: {fmt})")

    if fmt == "cooccurrence":
        data = pd.read_csv(hier_input)
        graph_builder = _build_graph_from_cooc_slice
    else:
        data = _load_and_normalise_triples(hier_input, tokens_path)
        graph_builder = _build_graph_from_triples_slice

    # Determine global sid range for death shading
    if sid_range is not None:
        srange = sid_range
    else:
        srange = (int(data["sentence_id"].min()), int(data["sentence_id"].max()))

    fmt_suffix = "_triples" if fmt == "triples" else ""

    for mode in modes:
        mode_suffix = f"_{mode}" if mode != "sentences" else ""
        suffix = f"{fmt_suffix}{mode_suffix}"
        hier_matrix, labels, edges, tp, tl, xlabel = _compute_hierarchy_matrix(
            data, graph_builder, victims, n_bins, tokens_path,
            bin_mode=mode, sid_range=sid_range,
        )
        print(f"\n[hierarchy / {fmt} / {mode}] {hier_matrix.shape[1]} bins")
        plot_hierarchy_timeline(hier_matrix, labels, edges, tp, tl, xlabel,
                                out_dir, suffix, sid_range=srange)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    chapter_table = format_chapter_ranges(TOKENS)

    description = (
        "Plot character mention timelines from co-occurrence data.\n\n"
        "Available chapter ranges (for --token-range):\n\n"
        f"{chapter_table}"
    )  

    from argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(
        description=description,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=Path,
        default=Path(f"{COOC_OUT}/raw_occurrences.csv"),
        help="Path to raw_occurrences.csv",
    )
    parser.add_argument(
        "-o", "--out", type=Path,
        default=Path(EGO_OUT) / "timeline",
        help="Output directory for the plots",
    )
    parser.add_argument(
        "-b", "--bins", type=int, default=50,
        help="Number of time bins (for 'sentences' and 'normalised' modes; default: 50)",
    )
    parser.add_argument(
        "-m", "--bin-mode",
        choices=["sentences", "normalised", "chapters", "all"],
        default="all",
        help=(
            "Binning mode: 'sentences' = raw sentence IDs, "
            "'normalised' = [0, 1] scale, "
            "'chapters' = one bin per chapter, "
            "'all' = produce all three (default)."
        ),
    )
    parser.add_argument(
        "-t", "--token-range", type=int_pair, default=None,
        help=(
            "Restrict to a token-ID range, e.g. '3780,7540'.\n"
            "Token IDs are mapped to sentence IDs via the tokens file.\n"
            "Use the chapter table above to pick chapter boundaries."
        ),
    )

    parser.add_argument(
        "-c", "--characters", type=str, default=None,
        help=(
            "Comma-separated character names to include (case-insensitive\n"
            "substring match). E.g. '-c Wargrave,Lombard,Brent'.\n"
            "If omitted, all main protagonists are plotted."
        ),
    )
    parser.add_argument(
        "--hierarchy", action="store_true",
        help=(
            "Plot each character's Burt hierarchy over time.\n"
            "Uses raw_cooccurrences.csv by default, or specify a\n"
            "triples TSV with --hierarchy-input."
        ),
    )
    parser.add_argument(
        "--hierarchy-input", type=Path, default=None,
        help=(
            "Path to input file for hierarchy computation.\n"
            "Accepts raw_cooccurrences.csv OR a triples TSV (avp/svo).\n"
            "Format is auto-detected. If omitted, uses\n"
            "raw_cooccurrences.csv from the cooccurrence output dir."
        ),
    )

    args = parser.parse_args()

    victims = load_victims(BASE_DATA_DIR, BASE_OUT_DIR)

    # ── Filter characters if requested ──
    if args.characters:
        queries = [q.strip().lower() for q in args.characters.split(",")]
        filtered = {}
        for cid, fullname in victims.items():
            name_lower = fullname.lower()
            if any(q in name_lower for q in queries):
                filtered[cid] = fullname
        if not filtered:
            print(f"No characters matched: {args.characters}")
            print(f"Available: {list(victims.values())}")
            return
        victims = filtered

    print(f"Characters ({len(victims)}): {list(victims.values())}")

    mentions, kind = build_mention_series(args.input, set(victims.keys()))
    print(f"Total mention records: {len(mentions)}")

    # ── Apply token-range filter ──
    sid_range = None
    if args.token_range:
        tok_lo, tok_hi = args.token_range
        sid_lo, sid_hi = token_range_to_sentence_range(tok_lo, tok_hi, TOKENS)
        sid_range = (sid_lo, sid_hi)
        mentions = mentions[
            (mentions["sentence_id"] >= sid_lo)
            & (mentions["sentence_id"] <= sid_hi)
        ].copy()
        print(f"Filtered to tokens [{tok_lo}, {tok_hi}] → sentences [{sid_lo}, {sid_hi}]")
        print(f"Mention records after filter: {len(mentions)}")

    modes = ["sentences", "chapters"] if args.bin_mode == "all" else [args.bin_mode] # "normalised", --> output equal to sentences

    for mode in modes:
        run_mode(mentions, victims, mode, args.bins, TOKENS, args.out, kind)

    # ── Hierarchy over time ──
    if args.hierarchy:
        if args.hierarchy_input is not None:
            hier_path = args.hierarchy_input
        else:
            hier_path = Path(COOC_OUT) / "raw_cooccurrences.csv"

        if not hier_path.exists():
            print(f"\n⚠ Hierarchy input not found: {hier_path}")
            print("  Supply a valid path via --hierarchy-input, or run")
            print("  cooccurrence.py first to generate raw_cooccurrences.csv.")
        else:
            run_hierarchy(hier_path, victims, args.bins, TOKENS, args.out,
                          modes, sid_range=sid_range)

    print("\nDone.")


if __name__ == "__main__":
    main()
