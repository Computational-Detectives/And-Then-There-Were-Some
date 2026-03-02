import os
import ast
import argparse
import pandas as pd

from pathlib import Path
from itertools import combinations
from collections import defaultdict
from typing import Dict, Set, Tuple, List

from ..config import TOKENS, ENTITY, BASE_OUT_DIR, COOC_OUT
from ..auxiliary import load_booknlp_file, print_headers, print_information


# ============================================================================
# MAPPING FUNCTIONS
# ============================================================================

def build_token_to_sentence_map(tokens_df: pd.DataFrame) -> Dict[int, int]:
    """
    Map each token_ID_within_document → sentence_ID.
    
    :param tokens_df: DataFrame loaded from .tokens file
    :return: Dict mapping token ID to sentence ID
    """
    return dict(zip(
        tokens_df["token_ID_within_document"].astype(int),
        tokens_df["sentence_ID"].astype(int)
    ))


def build_coref_to_canonical(canonical_df: pd.DataFrame) -> Dict[int, Tuple[int, str]]:
    """
    Map COREF ID → (canonical_id, fullname).
    
    Only includes entries with a valid match (avg_name_match_score != 0).
    
    :param canonical_df: DataFrame loaded from canonical_mappings.csv
    :return: Dict mapping COREF ID to (canonical_id, fullname) tuple
    """
    # Filter out unmatched entries
    canonical_df = canonical_df[canonical_df["avg_name_match_score"] != 0]
    
    coref_to_canonical = {}
    for _, row in canonical_df.iterrows():
        original_ids = ast.literal_eval(row["original_ids"])
        canonical_id = int(row["canonical_id"])
        fullname = row["fullname"]
        for orig_id in original_ids:
            coref_to_canonical[orig_id] = (canonical_id, fullname)
    
    return coref_to_canonical


# ============================================================================
# CHARACTER EXTRACTION
# ============================================================================

def extract_characters_per_sentence(
    entities_df: pd.DataFrame,
    token_to_sentence: Dict[int, int],
    coref_to_canonical: Dict[int, Tuple[int, str]]
) -> Dict[int, Set[int]]:
    """
    For each sentence, collect all unique canonical character IDs.
    
    Uses the .entities file which maps token ranges to COREF IDs,
    including pronouns like "He", "she", "him" that resolve to characters.
    
    :param entities_df: DataFrame loaded from .entities file
    :param token_to_sentence: Mapping from token ID to sentence ID
    :param coref_to_canonical: Mapping from COREF ID to canonical info
    :return: Dict mapping sentence_id to set of canonical character IDs
    """
    sentence_to_characters = defaultdict(set)
    
    for _, row in entities_df.iterrows():
        coref_id = row["COREF"]
        start_token = int(row["start_token"])
        
        # Get sentence for this entity mention
        sentence_id = token_to_sentence.get(start_token)
        if sentence_id is None:
            continue
        
        # Map COREF to canonical (skip if not a matched character)
        if coref_id in coref_to_canonical:
            canonical_id, _ = coref_to_canonical[coref_id]
            sentence_to_characters[sentence_id].add(canonical_id)
    
    return dict(sentence_to_characters)


# ============================================================================
# CO-OCCURRENCE GENERATION
# ============================================================================

def generate_cooccurrences(
    sentence_to_characters: Dict[int, Set[int]],
    coref_to_canonical: Dict[int, Tuple[int, str]]
) -> pd.DataFrame:
    """
    Generate all pairwise character co-occurrences per sentence.
    
    Only includes sentences with 2+ unique characters.
    Pairs are sorted (lower ID first) for consistency.
    
    :param sentence_to_characters: Mapping from sentence_id to character IDs
    :param coref_to_canonical: Mapping from COREF ID to canonical info
    :return: DataFrame with one row per sentence-level co-occurrence
    """
    records = []
    
    # Build canonical_id → fullname lookup (deduplicated)
    id_to_name = {}
    for canonical_id, fullname in coref_to_canonical.values():
        id_to_name[canonical_id] = fullname
    
    for sentence_id, char_ids in sentence_to_characters.items():
        if len(char_ids) < 2:
            continue
        
        # Generate all unique pairs (sorted to ensure consistency)
        for id1, id2 in combinations(sorted(char_ids), 2):
            records.append({
                "source_id": id1,
                "target_id": id2,
                "source_name": id_to_name.get(id1, ""),
                "target_name": id_to_name.get(id2, ""),
                "sentence_id": sentence_id
            })
    
    return pd.DataFrame(records)


def aggregate_edges(cooccurrences: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate co-occurrences into weighted edge list.
    
    :param cooccurrences: DataFrame with raw co-occurrences
    :return: DataFrame with aggregated edge weights
    """
    if cooccurrences.empty:
        return pd.DataFrame(columns=[
            "source_id", "target_id", "source_name", "target_name", 
            "weight", "sentence_ids"
        ])
    
    return (
        cooccurrences
        .groupby(["source_id", "target_id", "source_name", "target_name"])
        .agg(
            weight=("sentence_id", "count"),
            sentence_ids=("sentence_id", list)
        )
        .reset_index()
        .sort_values("weight", ascending=False)
    )


# ============================================================================
# RAW OCCURRENCE COUNTING
# ============================================================================

def count_raw_occurrences(
    sentence_to_characters: Dict[int, Set[int]],
    coref_to_canonical: Dict[int, Tuple[int, str]],
) -> pd.DataFrame:
    """
    Count per-character occurrences across sentences.

    For each character, records the total number of sentences they appear in
    and the list of sentence IDs.  This uses the same entity-resolution
    pipeline as the co-occurrence extraction (COREF → canonical mapping).

    :param sentence_to_characters: Mapping from sentence_id to character IDs
    :param coref_to_canonical: Mapping from COREF ID to canonical info
    :return: DataFrame sorted by count (descending) with columns
             canonical_id, fullname, count, sentence_ids
    """
    # Build canonical_id → fullname lookup
    id_to_name: Dict[int, str] = {}
    for canonical_id, fullname in coref_to_canonical.values():
        id_to_name[canonical_id] = fullname

    # Accumulate
    char_sentences: Dict[int, list] = defaultdict(list)
    for sentence_id, char_ids in sentence_to_characters.items():
        for cid in char_ids:
            char_sentences[cid].append(sentence_id)

    records = []
    for cid, sids in char_sentences.items():
        sids_sorted = sorted(sids)
        records.append({
            "canonical_id": cid,
            "fullname": id_to_name.get(cid, ""),
            "count": len(sids_sorted),
            "sentence_ids": sids_sorted,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("count", ascending=False).reset_index(drop=True)
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(output_dir: Path = COOC_OUT, verbose: bool = False, raw_occurrences: bool = True) -> None:
    print_headers("SENTENCE-LEVEL CO-OCCURRENCE EXTRACTION", "=", prefix="\n")
    
    # Resolve output paths
    canonical_path = output_dir.parent / "canonical_mappings.csv"
    
    # --------- Check prerequisites ---------
    if not canonical_path.exists():
        print_information(
            f"canonical_mappings.csv not found at {canonical_path}", 
            symb="!", col="RED"
        )
        print_information(
            "Please run match_names.py first to generate canonical mappings.",
            prefix="    "
        )
        print_information(
            "Example: python match_names.py -i ../out/preproc_attwn.book -o ../out",
            prefix="    "
        )
        return
    
    # --------- Load data files ---------
    print_information("Loading data files...", 1, prefix="\n")
    
    tokens_df = load_booknlp_file(TOKENS)
    print_information(f"Loaded {len(tokens_df)} tokens", prefix="    ")
    
    entities_df = load_booknlp_file(ENTITY)
    print_information(f"Loaded {len(entities_df)} entity mentions", prefix="    ")
    
    canonical_df = pd.read_csv(canonical_path)
    print_information(f"Loaded {len(canonical_df)} canonical mappings", prefix="    ")
    
    # --------- Build mappings ---------
    print_information("Building mappings...", 2, "\n")
    
    token_to_sentence = build_token_to_sentence_map(tokens_df)
    print_information(
        f"Built token→sentence map ({len(token_to_sentence)} tokens)", 
        prefix="    "
    )
    
    coref_to_canonical = build_coref_to_canonical(canonical_df)
    print_information(
        f"Built COREF→canonical map ({len(coref_to_canonical)} COREF IDs)", 
        prefix="    "
    )
    
    # --------- Extract characters per sentence ---------
    print_information("Extracting characters per sentence...", 3, "\n")
    
    sentence_chars = extract_characters_per_sentence(
        entities_df, token_to_sentence, coref_to_canonical
    )
    
    # Count sentences with 2+ characters
    multi_char_sentences = sum(1 for chars in sentence_chars.values() if len(chars) >= 2)
    print_information(
        f"Found {len(sentence_chars)} sentences with characters", 
        prefix="    "
    )
    print_information(
        f"Found {multi_char_sentences} sentences with 2+ characters",
        prefix="    "
    )
    
    # --------- Generate pairwise co-occurrences ---------
    print_information("Generating pairwise co-occurrences...", 4, "\n")
    
    cooccurrences = generate_cooccurrences(sentence_chars, coref_to_canonical)
    print_information(
        f"Generated {len(cooccurrences)} raw co-occurrence pairs",
        prefix="    "
    )
    
    # --------- Aggregate edges to get weights ---------
    print_information("Aggregating edges...", 5, "\n")
    
    edges = aggregate_edges(cooccurrences)
    print_information(
        f"Aggregated into {len(edges)} unique character pairs",
        prefix="    "
    )
    
    # --------- Save outputs ---------
    print_information("Saving outputs...", 6, "\n")
    
    if not output_dir.exists():
        os.makedirs(output_dir)
    
    raw_path = output_dir / "raw_cooccurrences.csv"
    cooccurrences.to_csv(raw_path, index=False)
    print_information(f"Raw co-occurrences saved to → {raw_path}", symb="✓")
    
    edges_path = output_dir / "edge_list.csv"
    edges.to_csv(edges_path, index=False)
    print_information(f"Edge list saved to → {edges_path}", symb="✓", col="GREEN")
    
    # --------- Raw occurrences (optional) ---------
    if raw_occurrences:
        print_information("Counting raw character occurrences...", 7, "\n")
        occ = count_raw_occurrences(sentence_chars, coref_to_canonical)
        occ_path = output_dir / "raw_occurrences.csv"
        occ.to_csv(occ_path, index=False)
        print_information(
            f"Raw occurrences ({len(occ)} characters) saved to → {occ_path}",
            symb="✓", col="GREEN",
        )

    # --------- Summary ---------
    if verbose:
        print_headers("CO-OCCURRENCE SUMMARY", "-", prefix="\n")
        
        print(f"    Total entity mentions processed:  {len(entities_df)}")
        print(f"    Sentences with characters:        {len(sentence_chars)}")
        print(f"    Sentences with 2+ characters:     {multi_char_sentences}")
        print(f"    Raw co-occurrence pairs:          {len(cooccurrences)}")
        print(f"    Unique character pairs:           {len(edges)}")
        
        if not edges.empty:
            print_headers("TOP 10 CHARACTER PAIRS", "-", prefix="\n")
            top_edges = edges.head(10)
            for _, row in top_edges.iterrows():
                print(f"    {row['source_name']} ↔ {row['target_name']}: {row['weight']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract sentence-level character co-occurrences."
    )
    parser.add_argument(
        "-o", "--out",
        default=COOC_OUT,
        type=Path,
        help="Output directory (must contain canonical_mappings.csv)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed statistics"
    )
    parser.add_argument(
        "--no-raw-occurrences",
        action="store_false",
        help="By default is also counts and saves raw per-character occurrences. Can be turned off with --no-raw-occurrences"
    )
    
    args = parser.parse_args()
    main(args.out, verbose=args.verbose, raw_occurrences=args.no_raw_occurrences)