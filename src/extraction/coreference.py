from __future__ import annotations

import json
import re

from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Set

from ..config import (
    CLEAN_NAMES, TOKENS, COREF_OUT, CHAR_RES_OUT,
    COREF_WINDOW_TOKENS, COREF_OVERLAP_TOKENS,
    FUZZY_AUTO_ACCEPT, RAW_TEXT, PRONOUNS
)

from .utils import (
    build_variant_index,
    print_headers, print_information, log_print,
    load_alias_dict, suppress_stdout,
    load_text, snap_span, setup_pipeline_logger
)

# ------- CONSTANTS -------
MALE_PRONOUNS   = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}

# Every surface form that must NEVER be stored as a text-keyed alias.
# Storing these literally causes them to be resolved globally in every
# subsequent window, independent of context — the primary cross-window
# contamination mechanism.
_BACKPROP_BLOCKLIST: Set[str] = {
    # 1st person singular
    "i", "me", "my", "mine", "myself",
    # Contracted 1st person forms that slip past the PRONOUNS check
    "i'm", "i'll", "i'd", "i've", "i'd", "i'll",
    # 2nd person
    "you", "your", "yours", "yourself", "yourselves",
    # 3rd person (already in PRONOUNS but listed explicitly for clarity)
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    # 1st person plural
    "we", "us", "our", "ours", "ourselves",
    # 3rd person plural
    "they", "them", "their", "theirs", "themselves",
    # Demonstratives / indefinites commonly grouped by the coref model
    "one", "this", "that", "these", "those", "who", "whom", "whose",
    # Address / vocative forms that are NOT name mentions
    "sir", "ma'am", "madam", "miss", "dear", "mister",
    # Generic nominals that fuzzy-match to characters via profession tokens
    "man", "woman", "fellow", "boy", "girl", "lady", "gentleman",
    "person", "figure", "creature", "body",
    # Discourse markers that Maverick occasionally clusters
    "here", "there", "now", "then",
}


# ============================================================================
# SAVE
# ============================================================================

def save_stage2(
    span_index: List[Dict[Any]],
    alias_dict: Dict[Any],
    unknown_clusters: List,
    out_dir: Path,
) -> Path:
    """Write Stage 2 outputs to disk."""
    stage_dir = out_dir / "coreference"
    stage_dir.mkdir(parents=True, exist_ok=True)

    with open(stage_dir / "span_index.jsonl", "w", encoding="utf-8") as f:
        for span in span_index:
            f.write(json.dumps(span, ensure_ascii=False) + "\n")

    with open(stage_dir / "alias_dict_extended.json", "w", encoding="utf-8") as f:
        json.dump(alias_dict, f, indent=2, ensure_ascii=False)

    with open(stage_dir / "unknown_clusters.json", "w", encoding="utf-8") as f:
        json.dump(unknown_clusters, f, indent=2, ensure_ascii=False)

    return stage_dir


# ============================================================================
# LOADING
# ============================================================================

def load_sentences(char_res_dir: Path) -> List[Dict[str, Any]]:
    """Reconstruct the sentence list from ner_spans.jsonl."""
    spans = []
    with open(char_res_dir / "ner_spans.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            spans.append(json.loads(line))
    return spans


def _load_coref_model(model_name: str = "maverick") -> Any:
    """Load the specified coref model."""
    if model_name == "maverick":
        import torch
        from maverick import Maverick

        original_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load

        model = Maverick(device="cpu", hf_name_or_path="sapienzanlp/maverick-mes-litbank")
        model.model = model.model.half()
        torch.load = original_torch_load

        from maverick.common.util import original_token_offsets
        import types

        @torch.no_grad()
        def _patched_predict(
            self, sample, singletons=False, add_gold_clusters=None,
            predefined_mentions=None, speakers=None
        ):
            tokens, eos_indices, speakers, char_offsets = self.preprocess(
                sample, speakers
            )
            tokenized = self.tokenize(
                tokens, eos_indices, speakers, predefined_mentions, add_gold_clusters
            )
            _dtype = next(self.model.parameters()).dtype
            output = self.model(
                stage="test",
                input_ids=torch.tensor(tokenized["input_ids"]).unsqueeze(0).to(self.device),
                attention_mask=torch.tensor(tokenized["attention_mask"])
                    .unsqueeze(0).to(self.device).to(_dtype),
                eos_mask=torch.tensor(tokenized["eos_mask"])
                    .unsqueeze(0).to(self.device).to(_dtype),
                tokens=[tokenized["tokens"]],
                subtoken_map=[tokenized["subtoken_map"]],
                new_token_map=[tokenized["new_token_map"]],
                singletons=singletons,
                add=tokenized["added"],
                gold_mentions=(
                    None
                    if tokenized["gold_mentions"] is None
                    else torch.tensor(
                        self.create_mention_matrix(
                            len(tokenized["input_ids"]),
                            tokenized["gold_mentions"],
                        )
                    ).unsqueeze(0).to(self.device)
                ),
            )
            clusters_predicted = original_token_offsets(
                clusters=output["pred_dict"]["clusters"],
                subtoken_map=tokenized["subtoken_map"],
                new_token_map=tokenized["new_token_map"],
            )
            result = {
                "tokens": tokens,
                "clusters_token_offsets": clusters_predicted,
                "clusters_char_offsets": None,
                "clusters_token_text": [
                    [" ".join(tokens[s[0]:s[1]+1]) for s in cluster]
                    for cluster in clusters_predicted
                ],
                "clusters_char_text": None,
            }
            if char_offsets is not None:
                result["clusters_char_offsets"] = [
                    [(char_offsets[s[0]][0], char_offsets[s[1]][1]) for s in cluster]
                    for cluster in clusters_predicted
                ]
            return result

        model.predict = types.MethodType(_patched_predict, model)
        return model

    elif model_name == "fastcoref":
        from fastcoref import FCoref
        return FCoref()
    else:
        raise ValueError(f"Unknown coref model: {model_name}")
    

# ============================================================================
# VERBOSE STATISTICS
# ============================================================================

def print_verbose_stats(
    all_resolved, alias_dict, extended_dict, clean_names, all_unknown,
    total_dropped=0
):
    from collections import Counter

    print_headers("STAGE 2 — VERBOSE STATISTICS", "-", prefix="\n")

    span_types = Counter()
    for s in all_resolved:
        text_lower = s["text"].lower().strip()
        if text_lower in PRONOUNS or text_lower in _BACKPROP_BLOCKLIST:
            span_types["pronoun"] += 1
        elif s["text"] in alias_dict:
            span_types["named_mention"] += 1
        else:
            span_types["nominal_reference"] += 1

    total_spans = len(all_resolved)
    log_print("    Resolved Span Types:")
    for stype, count in span_types.most_common():
        pct = count / total_spans * 100
        log_print(f"      {stype:20s} {count:5d}  ({pct:5.1f}%)")

    id_to_name = dict(zip(clean_names["id"], clean_names["fullname"]))
    char_resolved = Counter()
    for s in all_resolved:
        cid = s.get("canonical_id")
        if cid is not None:
            name = id_to_name.get(cid, f"char_{cid}")
            char_resolved[name] += 1

    log_print(f"\n    Per-Character Resolved Spans ({len(char_resolved)} characters):")
    for name, count in char_resolved.most_common():
        log_print(f"      {name:35s} {count:5d} spans")

    confidences = [s.get("confidence", 0) for s in all_resolved if s.get("confidence")]
    if confidences:
        log_print("\n    Confidence Distribution:")
        log_print(
            f"      Min: {min(confidences):.1f}  "
            f"Max: {max(confidences):.1f}  "
            f"Mean: {sum(confidences)/len(confidences):.1f}"
        )

    log_print("\n    Alias Dictionary Growth:")
    log_print(f"      Stage 1 entries:   {len(alias_dict)}")
    log_print(f"      Extended entries:  {len(extended_dict)}")
    log_print(f"      New entries:       {len(extended_dict) - len(alias_dict)}")

    if all_unknown:
        log_print(f"\n    Unknown Clusters ({len(all_unknown)}):")
        for i, cluster in enumerate(all_unknown[:5]):
            texts = [s["text"] for s in cluster]
            log_print(f"      Cluster {i+1}: {texts}")
        if len(all_unknown) > 5:
            log_print(f"      ... and {len(all_unknown) - 5} more")

    if total_dropped > 0:
        log_print("\n    Strict Dropping Statistics:")
        log_print(f"      Conflicting clusters fully/partially dropped: {total_dropped}")


# ============================================================================
# BACK-PROPAGATION
# ============================================================================

def _is_backprop_safe(text: str, base_alias_dict: Dict[Any]) -> bool:
    """
    Return True only if *text* is safe to store as a literal text key
    in the extended alias dict.

    A surface form is considered safe when:
      - It is already a known alias in the base (Stage 1) dict, OR
      - It contains at least one token that starts with a capital letter
        (i.e. it looks like a proper-name reference rather than a
        common-noun or pronoun mention).

    This blocks forms like "the old man", "dear", "you", "sir" from
    being written as global aliases while still allowing "The Judge",
    "Old Wargrave", or any surface form present in the seed dict.
    """
    if text in base_alias_dict:
        return True
    tokens = text.split()
    return any(t and t[0].isupper() for t in tokens)


def back_propagate(alias_dict: Dict[Any], resolved_spans: List[Dict[Any]]) -> Dict[Any]:
    """
    Extend the alias dict with newly resolved nominal references.

    Rules
    -----
    * Spans whose lower-cased text is in PRONOUNS or _BACKPROP_BLOCKLIST
      are always stored under span-indexed keys, never as literal text.
    * Spans that do not look like proper-name references (no capitalised
      token, not already in the base dict) are also stored under
      span-indexed keys rather than as global text aliases.
    * Only surface forms that pass _is_backprop_safe() are written as
      literal text keys — i.e. proper-name references like "The Judge"
      or "old Wargrave" that start with a capital somewhere.
    """
    extended = dict(alias_dict)

    for span in resolved_spans:
        text = span["text"]
        if span.get("canonical_id") is None:
            continue

        text_lower = text.lower().strip()
        is_pronoun_or_blocked = (
            text_lower in PRONOUNS or text_lower in _BACKPROP_BLOCKLIST
        )

        if is_pronoun_or_blocked or not _is_backprop_safe(text, alias_dict):
            key = f"__pronoun__{span['start_char']}_{span['end_char']}"
        else:
            key = text

        if key not in extended:
            extended[key] = {
                "canonical_id": span["canonical_id"],
                "fullname":     span.get("fullname", ""),
                "score":        span.get("confidence", 0.0),
            }

    return extended


# ============================================================================
# SCENE BOUNDARIES
# ============================================================================

def detect_scene_boundaries(text: str) -> List[int]:
    """
    Detect hard scene boundaries: Markdown chapter/section markers and
    'Chapter' text markers.

    :return: Sorted list of character offsets where new scenes begin.
    """
    boundaries: List[int] = []
    for m in re.finditer(r'(?m)^(#{1,2})\s+.*$', text):
        boundaries.append(m.start())
    return sorted(set(boundaries))


def _char_offset_to_sent_id(char_offset: int, sentences: List[Dict[Any]]) -> int:
    """Map a character offset to the sentence ID that contains it."""
    for sent in sentences:
        if sent["start_char"] <= char_offset < sent["end_char"]:
            return sent["sid"]
    return sentences[-1]["sid"] if sentences else 0


# ============================================================================
# WINDOWING LOGIC
# ============================================================================

def track_active_characters(
    resolved_spans: List[Dict[Any]],
    current_sid: int,
    decay_window: int = 30,
) -> Set[int]:
    """
    Return the set of canonical character IDs mentioned within
    the last *decay_window* sentences before *current_sid*.
    """
    active: Set[int] = set()
    for span in resolved_spans:
        if span.get("canonical_id") is not None:
            span_sid = span.get("sid", 0)
            if current_sid - decay_window <= span_sid <= current_sid:
                active.add(span["canonical_id"])
    return active


def build_windows(
    sentences: List[Dict[str, Any]],
    text: str,
    window_tokens: int = COREF_WINDOW_TOKENS,
    overlap_tokens: int = COREF_OVERLAP_TOKENS,
    scene_boundary_offsets: List[int] | None = None,
    prefer_scene_breaks: bool = True,
) -> List[Dict[str, Any]]:
    """
    Group sentences into overlapping windows, respecting scene boundaries.

    When ``prefer_scene_breaks==True`` (default) the window builder tries
    to end each window at the nearest scene break that still fits within
    the token budget, rather than cutting mid-scene.  This gives the coref
    model the maximum coherent context within a single chapter or section.
    If no scene break fits within the budget the window is closed at the
    token limit as before.

    :return: List of window dicts.
    """
    scene_sids: Set[int] = set()
    if scene_boundary_offsets:
        for offset in scene_boundary_offsets:
            scene_sids.add(_char_offset_to_sent_id(offset, sentences))

    i = 0
    window_id = 0
    windows: List[Dict[str, Any]] = []

    while i < len(sentences):
        window_sents: List[Dict[Any]] = []
        token_count = 0
        last_scene_break_j: Optional[int] = None  # last j that landed on a scene break

        j = i
        while j < len(sentences):
            sent = sentences[j]
            sent_tokens = len(sent.get("tokens", sent["text"].split()))

            if token_count + sent_tokens > window_tokens and len(window_sents) > 0:
                break

            token_count += sent_tokens
            window_sents.append(sent)

            # Record if the NEXT sentence opens a new scene
            if (j + 1) < len(sentences) and sentences[j + 1]["sid"] in scene_sids:
                last_scene_break_j = j + 1
                # If prefer_scene_breaks, stop the window here so it ends
                # cleanly at the chapter boundary
                if prefer_scene_breaks:
                    j += 1
                    break

            j += 1

        if window_sents:
            start_off = int(window_sents[0]["start_char"])
            end_off   = int(window_sents[-1]["end_char"])
            windows.append({
                "window_id": window_id,
                "sentences": window_sents,
                "text":      text[start_off:end_off],
                "start_sid": window_sents[0]["sid"],
                "end_sid":   window_sents[-1]["sid"],
                "tokens":    [t for s in window_sents for t in s.get("tokens", [])],
            })
            window_id += 1

        # Determine next window start
        if scene_sids and j < len(sentences) and sentences[j]["sid"] in scene_sids:
            next_start = j
        else:
            remaining_tokens = 0
            next_start = j
            for k in range(len(window_sents) - 1, -1, -1):
                remaining_tokens += len(
                    window_sents[k].get("tokens", window_sents[k]["text"].split())
                )
                if remaining_tokens >= overlap_tokens:
                    next_start = i + k
                    break

        if next_start <= i:
            next_start = i + 1
        i = next_start

        if j == len(sentences):
            break

    return windows


def build_window_header(
    active_chars: Set[int],
    names_df: pd.DataFrame,
) -> str:
    """
    Construct a scene context header for a coref window.

    :return: Header string like ``[SCENE CONTEXT: Active characters: ...]``.
    """
    if not active_chars:
        return ""

    id_to_name   = dict(zip(names_df["id"], names_df["fullname"]))
    id_to_gender = dict(zip(names_df["id"], names_df["gender"]))

    parts = []
    for cid in sorted(active_chars):
        name   = id_to_name.get(cid, f"char_{cid}")
        gender = id_to_gender.get(cid, "u")
        parts.append(f"{name} ({gender})")

    return f"[SCENE CONTEXT: Active characters: {', '.join(parts)}]"


# ============================================================================
# CLUSTER LOGIC
# ============================================================================

# =========== CLUSTER HELPERS ===========
def to_ontonotes_format_with_map(win: Dict[Any]):
    """
    Convert window sentences to OntoNotes word lists, splitting at quote
    tokens and discarding them.  Returns:
      - window_sents_words : List[List[str]] ready for Maverick
      - index_map          : dict mapping pseudo_token_idx → original token idx
    """
    quotes = {'"', '\u201c', '\u201d', "''", "``", "'"}
    window_sents_words = []
    index_map          = {}
    pseudo_idx         = 0
    orig_flat_idx      = 0

    for sent in win["sentences"]:
        sent_tokens        = sent.get("tokens", [])
        current_sent       = []
        current_orig_indices = []

        for t in sent_tokens:
            word = str(t["word"])
            if word in quotes:
                if current_sent:
                    window_sents_words.append(current_sent)
                    for local_i, orig_i in enumerate(current_orig_indices):
                        index_map[pseudo_idx + local_i] = orig_i
                    pseudo_idx += len(current_sent)
                    current_sent         = []
                    current_orig_indices = []
            else:
                current_sent.append(word)
                current_orig_indices.append(orig_flat_idx)
            orig_flat_idx += 1

        if current_sent:
            window_sents_words.append(current_sent)
            for local_i, orig_i in enumerate(current_orig_indices):
                index_map[pseudo_idx + local_i] = orig_i
            pseudo_idx += len(current_sent)

    return window_sents_words, index_map


def remap_clusters(clusters, index_map):
    """
    Remap cluster token offsets from pseudo-token space back to
    original token space, dropping spans with unmapped indices.
    """
    remapped = []
    for cluster in clusters:
        clean_cluster = [
            (index_map[start], index_map[end])
            for start, end in cluster
            if start in index_map and end in index_map
        ]
        if clean_cluster:
            remapped.append(clean_cluster)
    return remapped


# =========== CLUSTER MAIN ===========
def map_clusters_to_characters(
    clusters: List[Any],
    alias_dict: Dict[Any],
    names_df: pd.DataFrame,
    window_start_char: int = 0,
    tokens: List[Dict[Any]] = None,
    full_text: str = "",
    is_token_indexed: bool = False,
    active_char_ids: Optional[Set[int]] = None,   # IMPROVEMENT 4
) -> Tuple[List[Dict[Any]], List[List[Dict[Any]]], int]:
    """
    Map coref clusters to canonical characters using the alias dict.

    Improvements vs. original
    -------------------------
    * Uses confidence-weighted voting with outlier rejection (_elect_canonical_id).
    * Passes source text to match_name_improved() for the capitalisation guard.
    * Enforces the active-character constraint for fuzzy-only attributions:
      if a cluster has no named anchor and the fuzzy candidate is not in
      active_char_ids, the cluster is marked unknown rather than attributed.

    :return: ``(resolved_spans, unknown_clusters, num_dropped_clusters)``
    """
    variant_to_ids, all_variants, id_to_gender, all_name_tokens = build_variant_index(
        names_df
    )
    resolved_spans:   List[Dict[Any]]       = []
    unknown_clusters: List[List[Dict[Any]]] = []
    num_dropped_clusters               = 0

    for cluster in clusters:
        # ---- snap and globalise bounds ----
        snapped_cluster = []
        for span in cluster:
            if is_token_indexed:
                start_tok, end_tok = span
                final_start = int(tokens[start_tok]["byte_onset"])
                final_end   = int(tokens[end_tok]["byte_offset"])
                final_text  = full_text[final_start:final_end]
            else:
                global_start = window_start_char + span["start_char"]
                global_end   = window_start_char + span["end_char"]
                if tokens and full_text:
                    final_start, final_end, final_text = snap_span(
                        global_start, global_end, tokens, full_text
                    )
                else:
                    final_start, final_end, final_text = (
                        global_start, global_end, span["text"]
                    )
            snapped_cluster.append({
                "start_char": final_start,
                "end_char":   final_end,
                "text":       final_text,
            })
        cluster = snapped_cluster

        # ---- collect anchors via direct alias-dict lookup ----
        cluster_anchors = []
        fuzzy_only      = False   # True when the sole anchor came from fuzzy match

        for span in cluster:
            span_text = span["text"]

            if span_text in alias_dict:
                cluster_anchors.append({
                    "span":         span,
                    "canonical_id": alias_dict[span_text]["canonical_id"],
                    "fullname":     alias_dict[span_text]["fullname"],
                    "confidence":   alias_dict[span_text]["score"],
                })
                continue

            pronoun_key = f"__pronoun__{span['start_char']}_{span['end_char']}"
            if pronoun_key in alias_dict:
                cluster_anchors.append({
                    "span":         span,
                    "canonical_id": alias_dict[pronoun_key]["canonical_id"],
                    "fullname":     alias_dict[pronoun_key]["fullname"],
                    "confidence":   alias_dict[pronoun_key]["score"],
                })

        # ---- fuzzy fallback on longest span if no direct anchor ----
        if not cluster_anchors:
            longest_span = max(cluster, key=lambda s: len(s["text"]))
            cid, fname, score, _ = match_name(
                longest_span["text"], "u",
                variant_to_ids, all_variants, id_to_gender,
                names_df, all_name_tokens,
                threshold=FUZZY_AUTO_ACCEPT,
                source_text=longest_span["text"],
            )
            if cid is not None:
                # Active-character hard constraint
                # If we have no named anchor, only attribute to characters
                # who have appeared recently.  This prevents cold-start windows
                # from attributing pronoun-only clusters to characters not yet
                # (or no longer) on the scene.
                if active_char_ids is not None and cid not in active_char_ids:
                    # Fuzzy candidate is not active — mark as unknown
                    unknown_clusters.append([
                        {"text": s["text"],
                         "start_char": s["start_char"],
                         "end_char": s["end_char"]}
                        for s in cluster
                    ])
                    continue

                cluster_anchors.append({
                    "span":         longest_span,
                    "canonical_id": cid,
                    "fullname":     fname,
                    "confidence":   score,
                })
                fuzzy_only = True

        # ---- no anchor at all → unknown ----
        if not cluster_anchors:
            unknown_clusters.append([
                {"text": s["text"],
                 "start_char": s["start_char"],
                 "end_char": s["end_char"]}
                for s in cluster
            ])
            continue

        # ---- Confidence-weighted vote with outlier rejection ----
        from collections import Counter
        id_counts  = Counter(a["canonical_id"] for a in cluster_anchors)
        unique_ids = set(id_counts.keys())
        is_conflicting = len(unique_ids) > 1

        majority_id, _ = _elect_canonical_id(cluster_anchors)
        dropped_spans          = []
        cluster_dropped_partly = False

        for span in cluster:
            anchor_match = next(
                (a for a in cluster_anchors if a["span"] == span), None
            )

            if anchor_match:
                resolved_spans.append({
                    "text":         span["text"],
                    "start_char":   span["start_char"],
                    "end_char":     span["end_char"],
                    "canonical_id": int(anchor_match["canonical_id"]),
                    "fullname":     anchor_match["fullname"],
                    "confidence":   round(anchor_match["confidence"], 2),
                })
            else:
                span_lower = span["text"].lower().strip()

                if is_conflicting:
                    conflicting_genders = {id_to_gender.get(c, "u") for c in unique_ids}

                    if span_lower in MALE_PRONOUNS or span_lower in FEMALE_PRONOUNS:
                        p_gender      = "m" if span_lower in MALE_PRONOUNS else "f"
                        matching_cids = [
                            c for c in unique_ids
                            if id_to_gender.get(c, "u") == p_gender
                        ]
                        if len(matching_cids) == 1:
                            assigned_id = matching_cids[0]
                            info        = next(
                                a for a in cluster_anchors
                                if a["canonical_id"] == assigned_id
                            )
                            resolved_spans.append({
                                "text":         span["text"],
                                "start_char":   span["start_char"],
                                "end_char":     span["end_char"],
                                "canonical_id": int(assigned_id),
                                "fullname":     info["fullname"],
                                "confidence":   round(info["confidence"], 2),
                            })
                        else:
                            dropped_spans.append(span)
                            cluster_dropped_partly = True
                    else:
                        dropped_spans.append(span)
                        cluster_dropped_partly = True
                else:
                    # Clean cluster — assign to the elected majority
                    info = next(
                        a for a in cluster_anchors
                        if a["canonical_id"] == majority_id
                    )
                    resolved_spans.append({
                        "text":         span["text"],
                        "start_char":   span["start_char"],
                        "end_char":     span["end_char"],
                        "canonical_id": int(majority_id),
                        "fullname":     info["fullname"],
                        "confidence":   round(info["confidence"], 2),
                    })

        if cluster_dropped_partly:
            num_dropped_clusters += 1
            if dropped_spans:
                unknown_clusters.append([
                    {"text": s["text"],
                     "start_char": s["start_char"],
                     "end_char": s["end_char"]}
                    for s in dropped_spans
                ])

    return resolved_spans, unknown_clusters, num_dropped_clusters


# ============================================================================
# NAME MATCHING LOGIC
# ============================================================================

# =========== NAME MATCHING HELPERS ===========
def _composite_name_score(query: str, variant: str) -> float:
    """
    Score *query* against *variant* using a weighted combination of:
      - fuzz.WRatio        (strong on exact and partial matches)
      - a length penalty   (discourages very short queries matching long variants)

    Returns a score in [0, 100].
    """
    from rapidfuzz import fuzz

    base_score = fuzz.WRatio(query, variant)

    # Length penalty: if the query is much shorter than the variant, scale
    # down to avoid short tokens scoring spuriously high.
    len_ratio = len(query) / max(len(variant), 1)
    # penalty is 1.0 when lengths are equal, 0.7 when query is 30 % of variant
    length_penalty = 0.7 + 0.3 * min(len_ratio, 1.0)

    return base_score * length_penalty


def _elect_canonical_id(
    cluster_anchors: List[Dict[Any]],
) -> Tuple[Optional[int], float]:
    """
    Given a list of anchor dicts (each with 'canonical_id' and 'confidence'),
    elect the winning canonical_id using confidence-weighted voting.

    Outlier rejection: anchors whose confidence is more than 20 points
    below the mean cluster confidence are excluded before voting.  This
    prevents a single spurious fuzzy match from swinging the result.

    Returns (winning_id, total_weight_of_winner).
    """
    from collections import defaultdict

    if not cluster_anchors:
        return None, 0.0

    mean_conf = sum(a["confidence"] for a in cluster_anchors) / len(cluster_anchors)
    OUTLIER_THRESHOLD = 20.0

    valid_anchors = [
        a for a in cluster_anchors
        if (mean_conf - a["confidence"]) <= OUTLIER_THRESHOLD
    ]
    if not valid_anchors:
        valid_anchors = cluster_anchors  # keep all if all are outliers

    weights: Dict[int, float] = defaultdict(float)
    for a in valid_anchors:
        weights[a["canonical_id"]] += a["confidence"]

    winning_id  = max(weights, key=weights.__getitem__)
    return winning_id, weights[winning_id]


# =========== NAME MATCHING MAIN ===========
def match_name(
    name: str,
    gender: str,
    variant_to_ids: Dict[str, List[int]],
    all_variants: List[str],
    id_to_gender: Dict[int, str],
    names_df: pd.DataFrame,
    all_name_tokens: Set[str],
    threshold: float = 60.0,
    source_text: str = "",
) -> Tuple[Optional[int], str, float, Optional[str]]:
    """
    Improved drop-in replacement for match_name() from utils.py.

    Rules
    -----
    * Uses _composite_name_score (WRatio + length penalty) instead of
      token_sort_ratio, reducing false positives on short or common tokens.
    * The single-token fallback now requires:
        - the token to be longer than 3 characters, AND
        - the token to appear capitalised in the original source text
      This blocks profession / title tokens ("judge", "doctor", "captain")
      from matching, since they are present in all_name_tokens via the
      profession / aka columns but are not proper-name references.
    """
    # --- internal helpers ---
    def _normalize(n: str) -> str:
        from .utils import _normalize_name
        return _normalize_name(n)

    def _clean(n: str, tokens: Set[str]) -> str:
        from .utils import _clean_non_names
        return _clean_non_names(n, tokens)

    original_name = name
    normalized    = _normalize(name)
    cleaned       = _clean(normalized, all_name_tokens)

    if not cleaned:
        return None, original_name, 0.0, None

    from ..config import TITLES

    def _try_match(query: str) -> Tuple[Optional[int], Optional[str], float, Optional[str]]:
        # Score all variants with the composite scorer
        scored = [
            (variant, _composite_name_score(query, variant))
            for variant in all_variants
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        for variant, score in scored:
            if score < threshold:
                break

            candidate_ids = variant_to_ids[variant]
            for cid in candidate_ids:
                db_gender = id_to_gender.get(cid, "u")
                corrected_gender = None
                if gender != "u" and gender in TITLES.keys() and gender != db_gender:
                    penalty = 1.0 if gender == db_gender else 0.8
                    if (score * penalty) >= threshold:
                        corrected_gender = db_gender
                    else:
                        continue
                fullname = names_df.loc[names_df["id"] == cid, "fullname"].iloc[0]
                return cid, fullname, score, corrected_gender

        return None, None, 0.0, None

    # Full-phrase attempt
    cid, fullname, score, corrected = _try_match(cleaned)
    if cid is not None:
        return cid, fullname, score, corrected

    # Single-token fallback — only for capitalised tokens longer than 3 chars
    tokens = cleaned.split()
    for token in tokens:
        if len(token) <= 3:
            continue
        if token not in all_name_tokens:
            continue

        # Capitalisation check: the token must appear capitalised somewhere
        # in the original source text (i.e. it is a proper noun in context,
        # not a lower-cased profession / title).
        if source_text and token not in source_text:
            # Check if capitalised form is present
            capitalised = token.capitalize()
            if capitalised not in source_text:
                continue
        elif not source_text:
            # No source text provided — skip the fallback entirely to be safe
            continue

        cid, fullname, score, corrected = _try_match(token)
        if cid is not None:
            # Apply a conservative penalty for single-token matches
            return cid, fullname, score * 0.85, corrected

    return None, original_name, 0.0, None


# ============================================================================
# COREFERENCE MODEL
# ============================================================================

def run_coref(window_input: Any, model: Any, model_name: str = "maverick") -> List:
    """Run the coref model on a single window."""
    if model_name == "maverick":
        preds = model.predict(window_input)
        return preds.get("clusters_token_offsets", [])
    elif model_name == "fastcoref":
        preds = model.predict(texts=[window_input])
        clusters = []
        for cluster in preds[0].get_clusters(as_strings=False):
            c = [
                {"text": window_input[s:e], "start_char": s, "end_char": e}
                for s, e in cluster
            ]
            clusters.append(c)
        return clusters
    return []


def run_coref_pass(
    windows: List[Dict[Any]],
    already_resolved: List[Dict[Any]],
    alias_dict: Dict[Any],
    clean_names: pd.DataFrame,
    coref_model,
    coref_model_name: str,
    text: str,
    decay_window: int = 30,
) -> Tuple[List[Dict[Any]], List[List[Dict[Any]]], int]:
    """
    Run coreference resolution over all windows sequentially.

    Notes
    -----
    * `active_char_ids` is computed before each window and passed to
      `map_clusters_to_characters()` as a hard constraint on fuzzy-only
      attributions.
    * Window-level sentence IDs are tracked so `track_active_characters()`
      receives meaningful `current_sid` values.
    """
    all_resolved:  List[Dict[Any]]       = list(already_resolved)
    all_unknown:   List[List[Dict[Any]]] = []
    total_dropped                   = 0

    # Pre-build a seen set from already_resolved to avoid duplicates
    seen: Set[Tuple[int, int]] = {
        (s["start_char"], s["end_char"]) for s in all_resolved
    }

    for win in tqdm(windows):
        # ---- Compute active characters for this window ----
        current_sid    = win["start_sid"]
        active_char_ids = track_active_characters(
            all_resolved, current_sid, decay_window=decay_window
        )

        if coref_model_name == "maverick":
            window_sents_words, idx_map = to_ontonotes_format_with_map(win)
            clusters_token_idx          = run_coref(
                window_sents_words, coref_model, coref_model_name
            )
            clusters_token_idx = remap_clusters(clusters_token_idx, idx_map)

            resolved, unknown, dropped_count = map_clusters_to_characters(
                clusters_token_idx, alias_dict, clean_names,
                tokens=win.get("tokens", []),
                full_text=text,
                is_token_indexed=True,
                active_char_ids=active_char_ids,   # wired in
            )
            total_dropped += dropped_count

        # ---- Single dedup path, no extend() ----
        for span in resolved:
            key = (span.get("start_char", 0), span.get("end_char", 0))
            if key not in seen:
                all_resolved.append(span)
                seen.add(key)

        all_unknown.extend(unknown)

    return all_resolved, all_unknown, total_dropped


# ============================================================================
# MAIN
# ============================================================================

def main(
    text_path:        Path = RAW_TEXT,
    char_res_dir:     Path = CHAR_RES_OUT,
    names_csv:        Path = CLEAN_NAMES,
    out_dir:          Path = COREF_OUT,
    tokens_path:      Path = None,
    coref_model_name: str  = "maverick",
    refine:           bool = False,
    verbose:          bool = False,
    decay_window:     int  = 30,
    prefer_scene_breaks: bool = True,
) -> Path:
    """Stage 2 — Coreference Resolution"""
    from warnings import simplefilter
    simplefilter("ignore")

    log_path = out_dir / "pipeline.log"
    setup_pipeline_logger(log_path=log_path)

    print_headers("STAGE 2 — COREFERENCE RESOLUTION", "=", prefix="\n")

    # --------------------- LOAD INPUTS ---------------------
    print_information("Loading inputs...", 1, "\n")
    text        = load_text(text_path)
    alias_dict  = load_alias_dict(char_res_dir / "alias_dict.json")
    import pandas as pd
    clean_names = pd.read_csv(names_csv)
    print_information(f"Alias dict has {len(alias_dict)} entries", prefix="    ")

    # ------------------ SEGMENT SENTENCES ------------------
    print_information("Segmenting sentences from tokens file...", 2, "\n")
    from .character_resolution import segment_sentences
    sentences = segment_sentences(text, tokens_path)
    print_information(f"Found {len(sentences)} sentences", prefix="    ")

    # --------------- DETECT SCENE BOUNDARIES ---------------
    print_information("Detecting scene boundaries...", 3, "\n")
    scene_boundaries = detect_scene_boundaries(text)
    print_information(
        f"Found {len(scene_boundaries)} scene boundaries", prefix="    "
    )

    # -------------------- BUILD WINDOWS --------------------
    print_information("Building coref windows...", 4, "\n")
    windows = build_windows(
        sentences, text,
        scene_boundary_offsets=scene_boundaries,
        prefer_scene_breaks=prefer_scene_breaks,
    )
    print_information(f"Built {len(windows)} windows", prefix="    ")

    # ------------------ LOAD COREF MODEL -------------------
    print_information(f"Loading coref model '{coref_model_name}'...", 5, "\n")
    with suppress_stdout():
        coref_model = _load_coref_model(coref_model_name)
    print_information("Model loaded", prefix="    ")

    # ----------------- RUN COREF PER WINDOW ----------------
    print_information("Running coreference resolution...", 6, "\n")
    all_resolved, all_unknown, dropped = run_coref_pass(
        windows, [], alias_dict, clean_names,
        coref_model, coref_model_name, text,
        decay_window=decay_window,
    )

    print_information(f"Resolved {len(all_resolved)} spans", prefix="    ")
    print_information(f"Unknown clusters: {len(all_unknown)}", prefix="    ")

    # ---------------- BACK-PROPAGATE RESULTS ---------------
    print_information("Back-propagating to alias dictionary...", 7, "\n")
    extended_dict = back_propagate(alias_dict, all_resolved)
    new_entries   = len(extended_dict) - len(alias_dict)
    print_information(
        f"Added {new_entries} new entries to alias dict", prefix="    "
    )

    # ----------------- RUN REFINEMENT PASS -----------------
    if refine:
        print_information("Running refinement pass...", 8, "\n")
        all_resolved, all_unknown, dropped_refine = run_coref_pass(
            windows, all_resolved, extended_dict, clean_names,
            coref_model, coref_model_name, text,
            decay_window=decay_window,
        )
        extended_dict = back_propagate(extended_dict, all_resolved)
        dropped += dropped_refine
        print_information(
            f"After refinement: {len(all_resolved)} total resolved spans",
            prefix="    ",
        )
        print_information(
            f"After refinement: {len(all_unknown)} spans remain unresolved",
            prefix="    ",
        )

    # ----------------- PRINT VERBOSE STATS -----------------
    if verbose and all_resolved:
        print_verbose_stats(
            all_resolved, alias_dict, extended_dict, clean_names,
            all_unknown, dropped,
        )

    # --------------------- SAVE RESULTS --------------------
    print_information(
        "Saving Coreference outputs...",
        symb=(9 if refine else 8),
        prefix="\n",
    )
    out_dir = save_stage2(all_resolved, extended_dict, all_unknown, out_dir)
    print_information(f"Saved to → {out_dir}", "✓", col="GREEN")

    # --------------- VISUALISE COREF RESULTS ---------------
    from .viz_resolution import main as viz_main
    print_information(
        "Generating Coreference visualization...",
        symb=(10 if refine else 9),
        prefix="\n",
    )
    viz_main(
        text=text_path,
        spans_path=out_dir / "span_index.jsonl",
        unknown=out_dir / "unknown_clusters.json",
        out=out_dir / "coreference_visualization.html",
    )
    print_information(
        f"Saved to → {out_dir / 'coref_visualization.html'}", "✓", col="GREEN"
    )

    return out_dir


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: Coreference Resolution")
    parser.add_argument("--text-path",     type=Path, default=RAW_TEXT)
    parser.add_argument("--char-res-dir",  type=Path, default=CHAR_RES_OUT)
    parser.add_argument("--names-csv",     type=Path, default=CLEAN_NAMES)
    parser.add_argument("--out-dir",       type=Path, default=COREF_OUT)
    parser.add_argument("--tokens-path",   type=Path, default=TOKENS)
    parser.add_argument("--coref-model",   type=str,  default="maverick")
    parser.add_argument("--decay-window",  type=int,  default=30,
                        help="Sentences of recency for active-character tracking")
    parser.add_argument("--no-scene-align", action="store_true",
                        help="Disable chapter-aligned window boundaries")
    parser.add_argument("--refine",   action="store_true")
    parser.add_argument("-v", "--verbose",  action="store_true")

    args = parser.parse_args()

    main(
        text_path=args.text_path,
        char_res_dir=args.char_res_dir,
        names_csv=args.names_csv,
        out_dir=args.out_dir,
        tokens_path=args.tokens_path,
        coref_model_name=args.coref_model,
        refine=args.refine,
        verbose=args.verbose,
        decay_window=args.decay_window,
        prefer_scene_breaks=not args.no_scene_align,
    )