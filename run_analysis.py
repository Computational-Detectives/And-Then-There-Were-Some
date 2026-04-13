"""
run_analysis.py — Run analysis scripts (timeline & ego-network).

Usage (from the project root):
    python run_analysis.py timeline [options]
    python run_analysis.py ego [options]
    python run_analysis.py all [options]        # runs both sequentially

To see the arguments for each sub-command:
    python run_analysis.py timeline --help
    python run_analysis.py ego --help
"""
import sys
import argparse
from pathlib import Path

from src.config import (
    DATA_DIR, OUT_DIR, # COOC_OUT, 
    OBJ_OUT, EGO_OUT, TOKENS
)

from src.auxiliary import (
    int_pair, int_list,
    make_windows,
    format_chapter_ranges,
    token_range_to_sentence_range,
)

from src.extraction.utils import (
    print_information
)

# from src.analysis.timeline_data import (
#     load_victims as tl_load_victims,
#     build_mention_series,
# )

# from src.analysis.character_timeline import (
#     run_mode,
#     run_hierarchy,
# )

from src.analysis.ego_network import (
    load_victims as ego_load_victims,
    _detect_format, load_graph_from_triples,
    run_analysis, death_cutpoints,
    DEATH_INTERVALS
)

from src.analysis.ego_borda import (
    run as run_borda_ranking,
)

from src.analysis.ego_borda_viz import (
    run as visualise_borda_ranking,
)


# ============================================================
# --------------------- TIMELINE ARGS ------------------------
# ============================================================

def add_timeline_args(parser: argparse.ArgumentParser) -> None:
    """Register all character_timeline CLI arguments on *parser*."""
    parser.add_argument(
        "-i", "--input", type=Path,
        # default=Path(f"{COOC_OUT}/raw_occurrences.csv"),
        help="Path to raw_occurrences.csv or raw_cooccurrences.csv",
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
    parser.add_argument(
        "--weighted", action="store_true",
        help=(
            "Weight each mention by 1/(characters alive).\n"
            "Normalises for the shrinking cast over the narrative."
        ),
    )


def run_timeline(args: argparse.Namespace) -> None:
    """Execute the character-timeline analysis."""
    return
    victims = tl_load_victims(DATA_DIR, OUT_DIR)

    # Filter characters if requested
    if args.characters:
        queries = [q.strip().lower() for q in args.characters.split(",")]
        filtered = {}
        for cid, fullname in victims.items():
            if any(q in fullname.lower() for q in queries):
                filtered[cid] = fullname
        if not filtered:
            print(f"No characters matched: {args.characters}")
            print(f"Available: {list(victims.values())}")
            return
        victims = filtered

    print(f"Characters ({len(victims)}): {list(victims.values())}")

    mentions, kind = build_mention_series(args.input, set(victims.keys()))
    print(f"Total mention records: {len(mentions)}")

    # Apply token-range filter
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

    modes = (
        ["sentences", "chapters"]
        if args.bin_mode == "all"
        else [args.bin_mode]
    )

    for mode in modes:
        run_mode(mentions, victims, mode, args.bins, TOKENS, args.out, kind,
                 weighted=args.weighted)

    # Hierarchy over time
    if args.hierarchy:
        hier_path = (
            args.hierarchy_input
            if args.hierarchy_input is not None
            else Path("raw_cooccurrences.csv")
        )
        if not hier_path.exists():
            print(f"\n⚠ Hierarchy input not found: {hier_path}")
            print("  Supply a valid path via --hierarchy-input, or run")
            print("  cooccurrence.py first to generate raw_cooccurrences.csv.")
        else:
            run_hierarchy(
                hier_path, victims, args.bins, TOKENS, args.out,
                modes, sid_range=sid_range,
            )

    print("\nTimeline done.")


# ============================================================
# ------------------------ EGO ARGS --------------------------
# ============================================================

def add_ego_args(parser: argparse.ArgumentParser) -> None:
    """Register all ego_network CLI arguments on *parser*."""
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
        "-o", "--out", type=Path, default=Path(EGO_OUT),
        help="Output directory",
    )

    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="Only aggregate the first N windows (by window_lo order). Default: all."
    )
    
    parser.add_argument(
        "-t", "--cutpoints", type=int_list, default=None,
        help=(
            "Comma-separated cutpoints for temporal windowing, e.g. '5000,10000'.\n"
            "For triples these are token IDs (index column);\n"
            "for edge lists these are sentence IDs.\n"
            "(Default: Split into Death Intervals)"
        ),
    )
    parser.add_argument(
        "--include-new-triples", action="store_true",
        help=(
            "Include manually added triples."
        ),
    )

    parser.add_argument(
        "-k", "--min-overlap", type=int, default=2,
        help="Minimum victim count for an alter to be flagged (default: 2)",
    )

    parser.add_argument(
        "-w", "--web", action="store_true",
        help="Generate interactive pyvis HTML visualisations",
    )


def run_ego(args: argparse.Namespace) -> None:
    """Execute the ego-network analysis."""
    fmt = args.format or _detect_format(args.input)
    print(f"Input : {args.input}")
    print(f"Format: {fmt}")

    # Load data
    if fmt == "triples":
        df_raw = load_graph_from_triples(args.input, include_new_triples=args.include_new_triples)

        # Exclude last two Epilogue & Manuscript
        df_raw = df_raw[df_raw['index'] <= 60964]

        temporal_col_min = DEATH_INTERVALS[0].token_start
        temporal_col_max = DEATH_INTERVALS[-1].token_end

    # Load victims
    victims = ego_load_victims(DATA_DIR)
    print(f"Victims ({len(victims)}): {list(victims.values())}")

    # Build windows
    # If no cutpoints supplied, derive them from the death schedule so that
    # window boundaries are guaranteed consistent with DEATH_INTERVALS.
    cutpoints = args.cutpoints if args.cutpoints is not None else death_cutpoints(fmt)
    windows = make_windows(cutpoints, temporal_col_min, temporal_col_max)
    print(f"Windows: {windows}\n")

    # Run Analysis
    run_analysis(
        df_raw, fmt, victims, windows,
        args.min_overlap, args.out, args.web,
    )

    # Run Borda Ranking
    run_borda_ranking(per_window_path=args.out, output_dir=args.out / "borda", max_windows=args.max_windows)

    # Visualise Borda Ranking
    visualise_borda_ranking(
        per_window_path=args.out / "borda" / "borda_per_window.csv",
        aggregated_path=args.out / "borda" / "borda_aggregated.csv",
        output_dir=args.out / "borda" / "visuals",
    )
    
    print("\nEgo-network done.")


# ============================================================
# -------------------------- MAIN ----------------------------
# ============================================================

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(
        description="Run analysis scripts (character timeline & ego-network).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = main_parser.add_subparsers(dest="command", help="Analysis to run")

    parser = argparse.ArgumentParser(description="Run Ego-Network Analysis on Extracted Triples")

    # --- timeline ---
    chapter_table = format_chapter_ranges(TOKENS)

    timeline_description = (
        "Plot character mention timelines from co-occurrence data.\n\n"
        "Available chapter ranges (for --token-range):\n\n"
        f"{chapter_table}"
    )
    tl_parser = subparsers.add_parser(
        name="timeline",
        description=timeline_description, # "timeline",
        help="Plot character mention timelines",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_timeline_args(tl_parser)

    # --- ego ---
    ego_parser = subparsers.add_parser(
        "ego",
        help="Ego-network analysis for perpetrator detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_ego_args(ego_parser)

    # --- all ---
    all_parser = subparsers.add_parser(
        "all",
        help="Run both timeline and ego-network analyses",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Run both timeline and ego-network analyses.\n\n"
            "The input file (-i) is a triples TSV (e.g. combined_triples.csv).\n"
            "It is used as:\n"
            "  • ego-network input\n"
            "  • timeline hierarchy input (--hierarchy is auto-enabled)\n\n"
            "The temporal occurrence plots always use raw_occurrences.csv\n"
            "(or raw_cooccurrences.csv) from the cooccurrence output dir."
        ),
    )
    all_parser.add_argument(
        "-i", "--input", type=Path,
        default=Path(OUT_DIR) / "triples.csv",
        help="Path to triples TSV — used for ego-network and hierarchy",
    )

    all_parser.add_argument(
        "-o", "--out", type=Path, default=Path(EGO_OUT),
        help="Base output directory (ego writes to <out>/ego_network)",
    )

    # Ego-specific
    all_parser.add_argument(
        "-f", "--format", choices=["triples", "edges"], default=None,
        help="Ego input format. Auto-detected if omitted.",
    )
    all_parser.add_argument(
        "-k", "--min-overlap", type=int, default=2,
        help="Ego: minimum victim count for alter overlap (default: 2)",
    )

    all_parser.add_argument(
        "-w", "--web", action="store_true",
        help="Ego: generate interactive pyvis HTML visualisations",
    )
    
    all_parser.add_argument(
        "--include-new-triples", action="store_true", default=False,
        help=(
            "Include manually added triples. (Default=False)"
        )
    )

    all_parser.add_argument(
        "--cutpoints", type=int_list, default=None,
        help="Ego: comma-separated temporal cutpoints, e.g. '5000,10000'",
    )


    # Timeline-specific
    # all_parser.add_argument(
    #     "-b", "--bins", type=int, default=50,
    #     help="[!] NOT AVAILABLE: Timeline: number of time bins (default: 50)",
    # )
    # all_parser.add_argument(
    #     "-m", "--bin-mode",
    #     choices=["sentences", "normalised", "chapters", "all"],
    #     default="all",
    #     help="[!] NOT AVAILABLE: Timeline: binning mode (default: all)",
    # )
    # all_parser.add_argument(
    #     "-t", "--token-range", type=int_pair, default=None,
    #     help="[!] NOT AVAILABLE: Timeline: restrict to token-ID range, e.g. '3780,7540'",
    # )
    # all_parser.add_argument(
    #     "-c", "--characters", type=str, default=None,
    #     help="[!] NOT AVAILABLE: Timeline: comma-separated character name filter",
    # )
    # all_parser.add_argument(
    #     "--timeline-input", type=Path,
    #     # default=Path(f"{COOC_OUT}/raw_occurrences.csv"),
    #     help="[!] NOT AVAILABLE: Timeline: occurrence input file (default: raw_occurrences.csv)",
    # )
    # all_parser.add_argument(
    #     "--weighted", action="store_true",
    #     help="[!] NOT AVAILABLE: Timeline: weight mentions by 1/(characters alive)",
    # )
    # 
    # all_parser.add_argument(
    #     "--hierarchy", action="store_true", default=True,
    #     help="[!] NOT AVAILABLE: (Always on in 'all' mode — accepted for compatibility)",
    # )
    # all_parser.add_argument(
    #     "--hierarchy-input", type=Path, default=None,
    #     help="[!] NOT AVAILABLE: Override hierarchy input (defaults to the -i triples file)",
    # )

    args = main_parser.parse_args()

    if args.command is None:
        main_parser.print_help()
        sys.exit(1)
    elif args.command == "ego":
        run_ego(args)
    elif args.command == "timeline":
        # run_timeline(args)
        print_information("Timeline analysis currently not available. Please select 'ego' instead.", symb='!', col="YELLOW")
        exit(1)
    elif args.command == "all":
        print_information("Timeline analysis currently not available. Please select 'ego' instead.", symb='!', col="YELLOW")
        exit(1)
        # Build namespaces for each sub-analysis from the shared args.
        # Timeline: occurrence plots use --timeline-input (default raw_occurrences),
        #           hierarchy uses --hierarchy-input if given, else the -i triples file.
        hier_file = args.hierarchy_input if args.hierarchy_input is not None else args.input
        tl_args = argparse.Namespace(
            input=args.timeline_input,
            out=args.out / "timeline",
            bins=args.bins,
            bin_mode=args.bin_mode,
            token_range=args.token_range,
            characters=args.characters,
            hierarchy=True,                 # auto-enabled
            hierarchy_input=hier_file,
            weighted=args.weighted,
        )

        # Ego: uses the -i triples file directly.
        ego_args = argparse.Namespace(
            input=args.input,
            format=args.format,
            min_overlap=args.min_overlap,
            out=args.out,
            web=args.web,
            cutpoints=args.cutpoints,
            include_new_triples=args.include_new_triples,
        )

        print("=" * 60)
        print("TIMELINE")
        print("=" * 60)
        # run_timeline(tl_args)
        print("\n")
        print("=" * 60)
        print("EGO-NETWORK")
        print("=" * 60)
        run_ego(ego_args)
