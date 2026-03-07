"""
Data loading, binning, and graph-building helpers for character-timeline analysis.

Functions:
    load_victims                        – Load main protagonist mapping
    build_mention_series_from_cooccurrences – Mentions from pairwise co-occurrences
    build_mention_series_from_occurrences  – Mentions from per-character occurrences
    build_mention_series                – Auto-detect format and load mentions
    _bin_equal                          – Equal-width bins over raw sentence IDs
    _bin_normalised                     – Equal-width bins, x-axis normalised to [0,1]
    _bin_chapters                       – One bin per chapter
    _build_matrix                       – Build character × bin count matrix
    _detect_hierarchy_format            – Detect cooccurrence vs triples file
    _load_and_normalise_triples         – Load + normalise triples TSV
    _build_graph_from_cooc_slice        – Build nx.Graph from co-occurrence slice
    _build_graph_from_triples_slice     – Build nx.Graph from triples slice
"""
from __future__ import annotations

import os

from pathlib import Path

from ..config import BASE_DATA_DIR, BASE_OUT_DIR
from ..auxiliary import build_chapter_sentence_ranges


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_victims(base_data_dir: str, base_out_dir: str) -> dict[int, str]:
    """
    Return {canonical_id: fullname} for main protagonists.
    """
    import pandas as pd
    names = pd.read_csv(os.path.join(base_data_dir, "names_owen_split.csv"))
    canonical = pd.read_csv(os.path.join(base_out_dir, "canonical_mappings.csv"))
    main_names = set(names.loc[names["prot_status"] == "main", "fullname"])
    victims = canonical[canonical["fullname"].isin(main_names)]
    return dict(zip(victims["canonical_id"], victims["fullname"]))


def build_mention_series_from_cooccurrences(
    raw_path: Path, character_ids: set[int]
) -> pd.DataFrame:
    """
    From raw_cooccurrences.csv, extract per-character mention records.

    A character is 'mentioned' in a sentence if it appears as either source
    or target.  Deduplicated so each (char_id, sentence_id) counts once.
    """
    import pandas as pd
    df = pd.read_csv(raw_path)

    src = df[df["source_id"].isin(character_ids)][
        ["source_id", "source_name", "sentence_id"]
    ].rename(columns={"source_id": "char_id", "source_name": "char_name"})

    tgt = df[df["target_id"].isin(character_ids)][
        ["target_id", "target_name", "sentence_id"]
    ].rename(columns={"target_id": "char_id", "target_name": "char_name"})

    mentions = pd.concat([src, tgt], ignore_index=True)
    mentions = mentions.drop_duplicates(subset=["char_id", "sentence_id"])
    return mentions


def build_mention_series_from_occurrences(
    raw_path: Path, character_ids: set[int]
) -> pd.DataFrame:
    """
    From raw_occurrences.csv, extract per-character mention records.

    Each row has canonical_id, fullname, count, sentence_ids (a list).
    We explode the sentence_ids into individual (char_id, sentence_id) rows.
    """
    import ast
    import pandas as pd
    df = pd.read_csv(raw_path)
    df = df[df["canonical_id"].isin(character_ids)].copy()

    # sentence_ids is stored as a string repr of a list
    df["sentence_ids"] = df["sentence_ids"].apply(ast.literal_eval)
    df = df.explode("sentence_ids").rename(columns={
        "canonical_id": "char_id",
        "fullname": "char_name",
        "sentence_ids": "sentence_id",
    })
    df["sentence_id"] = df["sentence_id"].astype(int)
    return df[["char_id", "char_name", "sentence_id"]]


def build_mention_series(
    raw_path: Path, character_ids: set[int]
) -> tuple[pd.DataFrame, str]:
    """
    Auto-detect input format and load mention records.

    Supports:
      - raw_cooccurrences.csv  (columns: source_id, target_id, …, sentence_id)
      - raw_occurrences.csv    (columns: canonical_id, fullname, count, sentence_ids)
    """
    # Peek at columns to auto-detect
    import pandas as pd
    cols = set(pd.read_csv(raw_path, nrows=0).columns)
    if {"source_id", "target_id", "sentence_id"} <= cols:
        print("  Format detected: raw_cooccurrences (pairwise)")
        return build_mention_series_from_cooccurrences(raw_path, character_ids), "cooccurrences"
    elif {"canonical_id", "sentence_ids"} <= cols:
        print("  Format detected: raw_occurrences (per-character)")
        return build_mention_series_from_occurrences(raw_path, character_ids), "occurrences"
    else:
        raise ValueError(
            f"Unrecognised input format. Columns found: {cols}\n"
            "Expected raw_cooccurrences.csv or raw_occurrences.csv"
        )


# ─────────────────────────────────────────────
# BINNING
# ─────────────────────────────────────────────

def _bin_equal(mentions: pd.DataFrame, n_bins: int):
    """
    Equal-width bins over raw sentence IDs.
    Returns (mentions_with_bin_col, bin_edges, bin_labels, xlabel).
    """
    min_s = mentions["sentence_id"].min()
    max_s = mentions["sentence_id"].max()
    import numpy as np
    import pandas as pd
    edges = np.linspace(min_s, max_s + 1, n_bins + 1)
    mentions = mentions.copy()
    mentions["bin"] = pd.cut(
        mentions["sentence_id"], bins=edges, labels=False, include_lowest=True
    )
    # Tick labels = sentence IDs at bin starts
    n_ticks = min(n_bins, 20)
    tick_pos = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
    tick_lab = [f"{int(edges[t])}" for t in tick_pos]
    return mentions, n_bins, tick_pos, tick_lab, edges, "Sentence ID (narrative progression →)"


def _bin_normalised(mentions: pd.DataFrame, n_bins: int):
    """
    Equal-width bins, but x-axis is compressed to [0, 1].
    """
    min_s = mentions["sentence_id"].min()
    max_s = mentions["sentence_id"].max()
    import numpy as np
    import pandas as pd
    raw_edges = np.linspace(min_s, max_s + 1, n_bins + 1)
    mentions = mentions.copy()
    mentions["bin"] = pd.cut(
        mentions["sentence_id"], bins=raw_edges, labels=False, include_lowest=True
    )
    n_ticks = min(n_bins, 20)
    tick_pos = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
    tick_lab = [f"{t / (n_bins - 1):.2f}" for t in tick_pos]
    # Normalise edges to [0, 1] so stream/line charts use that scale
    norm_edges = np.linspace(0.0, 1.0, n_bins + 1)
    return mentions, n_bins, tick_pos, tick_lab, norm_edges, "Narrative progression (0 = start, 1 = end)"


def _bin_chapters(mentions: pd.DataFrame, tokens_path: str):
    """
    One bin per chapter, derived from token → sentence ID ranges.
    Only chapters that overlap with the current mention data are included.
    """
    import numpy as np
    all_chapters = build_chapter_sentence_ranges(tokens_path)

    # Keep only chapters that overlap with the mention sentence range
    min_sid = mentions["sentence_id"].min()
    max_sid = mentions["sentence_id"].max()
    chapters = [
        ch for ch in all_chapters
        if ch["end_sid"] >= min_sid and ch["start_sid"] <= max_sid
    ]
    n_bins = len(chapters)

    mentions = mentions.copy()
    mentions["bin"] = -1
    for i, ch in enumerate(chapters):
        mask = (
            (mentions["sentence_id"] >= ch["start_sid"])
            & (mentions["sentence_id"] <= ch["end_sid"])
        )
        mentions.loc[mask, "bin"] = i

    mentions = mentions[mentions["bin"] >= 0]

    tick_pos = np.arange(n_bins)
    tick_lab = [f"Ch {ch['chapter']}" for ch in chapters]
    edges = None  # not used for stream-chart centres; handled separately
    return mentions, n_bins, tick_pos, tick_lab, edges, "Chapter"


def _build_matrix(
    mentions: pd.DataFrame, char_ids: list[int], n_bins: int
) -> np.ndarray:
    import numpy as np
    matrix = np.zeros((len(char_ids), n_bins), dtype=int)
    for i, cid in enumerate(char_ids):
        counts = mentions[mentions["char_id"] == cid].groupby("bin").size()
        for b, cnt in counts.items():
            if b is not None and 0 <= int(b) < n_bins:
                matrix[i, int(b)] = cnt
    return matrix


# ─────────────────────────────────────────────
# HIERARCHY DATA HELPERS
# ─────────────────────────────────────────────

def _detect_hierarchy_format(path: Path) -> str:
    """Detect whether a hierarchy input file is cooccurrence or triples."""
    with open(path) as f:
        header = f.readline()
    if "source_id" in header and "sentence_id" in header:
        return "cooccurrence"
    return "triples"


def _build_graph_from_cooc_slice(cooc_slice: pd.DataFrame) -> "nx.Graph":
    """Build a weighted undirected graph from a slice of raw_cooccurrences."""
    import networkx as nx

    agg = (
        cooc_slice
        .groupby(["source_id", "target_id", "source_name", "target_name"])
        .size()
        .reset_index(name="weight")
    )
    G = nx.Graph()
    for _, row in agg.iterrows():
        G.add_edge(row["source_id"], row["target_id"], weight=row["weight"])
        G.nodes[row["source_id"]].setdefault("name", row["source_name"])
        G.nodes[row["target_id"]].setdefault("name", row["target_name"])
    return G


def _build_graph_from_triples_slice(triples_slice: pd.DataFrame) -> "nx.Graph":
    """Build a weighted undirected graph from a slice of AVP/SVO triples."""
    import networkx as nx

    # Remove self-loops
    slc = triples_slice[
        triples_slice["canonical_id_left"] != triples_slice["canonical_id_right"]
    ]
    if slc.empty:
        return nx.Graph()

    agg = (
        slc.groupby(["canonical_id_left", "canonical_id_right",
                     "name_left", "name_right"])
        .size()
        .reset_index(name="weight")
    )
    G = nx.Graph()
    for _, row in agg.iterrows():
        src, tgt = int(row["canonical_id_left"]), int(row["canonical_id_right"])
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += row["weight"]
        else:
            G.add_edge(src, tgt, weight=row["weight"])
        G.nodes[src].setdefault("name", row["name_left"])
        G.nodes[tgt].setdefault("name", row["name_right"])
    return G


def _load_and_normalise_triples(path: Path, tokens_path: str) -> pd.DataFrame:
    """
    Load a triples TSV file, normalise column names (SVO → AVP style),
    and map the verb/action token ID to a sentence ID.

    Handles both SVO (``verb_id``) and AVP/combined (``index``) token-ID
    column names.
    """
    import pandas as pd
    df = pd.read_csv(path, sep="\t")

    # Normalise SVO column names to match AVP / combined names
    col_map = {
        "canonical_subj_id": "canonical_id_left",
        "canonical_obj_id": "canonical_id_right",
        "subject_text": "name_left",
        "object_text": "name_right",
        "verb_id": "index",            # SVO uses verb_id; AVP/combined use index
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
              inplace=True)

    # Map token IDs → sentence IDs
    tokens = pd.read_csv(tokens_path, sep="\t",
                         usecols=["token_ID_within_document", "sentence_ID"])
    tok2sid = dict(zip(tokens["token_ID_within_document"],
                       tokens["sentence_ID"]))
    df["sentence_id"] = df["index"].map(tok2sid)
    df = df.dropna(subset=["sentence_id"])
    df["sentence_id"] = df["sentence_id"].astype(int)

    return df
