"""
Stage 2 — Coreference Resolution

Resolve pronouns, possessives, and nominal references to canonical
character IDs using a neural coref model (Maverick or FastCoref),
anchored by the Stage 1 alias dictionary.
"""
from __future__ import annotations

import json
import re
import pandas as pd
import os

from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Dict, Tuple

from .config import (
    CLEAN_NAMES, TOKENS, COREF_OUT, CHAR_RES_OUT, 
    COREF_WINDOW_TOKENS, COREF_OVERLAP_TOKENS,
    FUZZY_AUTO_ACCEPT, RAW_TEXT, PRONOUNS 
)

MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}

from .utils import (
    build_variant_index, match_name,
    print_headers, print_information,
    load_alias_dict, suppress_stdout,
    load_text, snap_span
)


# TODO: If multiple people in cluster, check if pronoun is plural --> use all resolved names as characters

# =============================================================
# --------------------- VERBOSE STATISTICS --------------------
# =============================================================
def print_verbose_stats(all_resolved, alias_dict, extended_dict, clean_names, all_unknown, total_dropped=0):
    from collections import Counter

    print_headers("STAGE 2 — VERBOSE STATISTICS", "-", prefix="\n")

    # Span type breakdown (pronoun vs named vs nominal)
    span_types = Counter()
    for s in all_resolved:
        text_lower = s["text"].lower().strip()
        if text_lower in PRONOUNS:
            span_types["pronoun"] += 1
        elif s["text"] in alias_dict:
            span_types["named_mention"] += 1
        else:
            span_types["nominal_reference"] += 1

    total_spans = len(all_resolved)
    print("    Resolved Span Types:")
    for stype, count in span_types.most_common():
        pct = count / total_spans * 100
        print(f"      {stype:20s} {count:5d}  ({pct:5.1f}%)")

    # Per-character resolution counts
    id_to_name = dict(zip(clean_names["id"], clean_names["fullname"]))
    char_resolved = Counter()
    for s in all_resolved:
        cid = s.get("canonical_id")
        if cid is not None:
            name = id_to_name.get(cid, f"char_{cid}")
            char_resolved[name] += 1

    print(f"\n    Per-Character Resolved Spans ({len(char_resolved)} characters):")
    for name, count in char_resolved.most_common():
        print(f"      {name:35s} {count:5d} spans")

    # Confidence distribution
    confidences = [s.get("confidence", 0) for s in all_resolved if s.get("confidence")]
    if confidences:
        print("\n    Confidence Distribution:")
        print(f"      Min: {min(confidences):.1f}  Max: {max(confidences):.1f}  Mean: {sum(confidences)/len(confidences):.1f}")

    # Alias dict growth
    print("\n    Alias Dictionary Growth:")
    print(f"      Stage 1 entries:   {len(alias_dict)}")
    print(f"      Extended entries:  {len(extended_dict)}")
    print(f"      New entries:       {len(extended_dict) - len(alias_dict)}")

    # Unknown clusters summary
    if all_unknown:
        print(f"\n    Unknown Clusters ({len(all_unknown)}):")
        for i, cluster in enumerate(all_unknown[:5]):
            texts = [s["text"] for s in cluster]
            print(f"      Cluster {i+1}: {texts}")
        if len(all_unknown) > 5:
            print(f"      ... and {len(all_unknown) - 5} more")
            
    if total_dropped > 0:
        print("\n    Strict Dropping Statistics:")
        print(f"      Conflicting clusters fully/partially dropped: {total_dropped}")


# ============================================================================
# LOADING
# ============================================================================

def load_sentences(char_res_dir: Path) -> List[Dict[str, Any]]:
    """
    Reconstruct the sentence list from ner_spans.jsonl.
    Each sentence gets its text from the span records.
    """
    spans = []
    with open(char_res_dir / "ner_spans.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            spans.append(json.loads(line))
    return spans


# ============================================================================
# SCENE BOUNDARIES
# ============================================================================

def detect_scene_boundaries(text: str) -> List[int]:
    """
    Detect hard scene boundaries: Markdown chapter/section markers (# and ##),
    'Chapter' text markers, '***', '---'.

    :return: List of character offsets where new scenes begin.
    """
    boundaries: List[int] = []

    # Markdown chapter markers (# Chapter... or ## Section...)
    for m in re.finditer(r'(?m)^(#{1,2})\s+.*$', text): # \s*
        boundaries.append(m.start()) # , m.group()))

    # Chapter markers
    # for m in re.finditer(r'\b(Chapter|Section)\s+\d+', text, re.IGNORECASE):
    #     boundaries.append((m.start(), m.group()))

    # Epilogue / Manuscript markers
    # for m in re.finditer(r'\b(Epilogue|MANUSCRIPT)\b', text):
    #     boundaries.append((m.start(), m.group()))

    # Separator lines
    # for m in re.finditer(r'\n\s*(\*\*\*|---)\s*\n', text):
    #     boundaries.append(m.start())

    return sorted(set(boundaries)) # sorted(set(boundaries), key=lambda x: x[0]) # 


def _char_offset_to_sent_id(char_offset: int, sentences: list[dict]) -> int:
    """Map a character offset to the sentence ID that contains it."""
    for sent in sentences:
        if sent["start_char"] <= char_offset < sent["end_char"]:
            return sent["sid"]
    return sentences[-1]["sid"] if sentences else 0


# ============================================================================
# WINDOWING
# ============================================================================

def build_windows(
    sentences: List[Dict[str, Any]],
    text: str,
    window_tokens: int = COREF_WINDOW_TOKENS,
    overlap_tokens: int = COREF_OVERLAP_TOKENS,
    scene_boundary_offsets: List[int] | None = None,
) -> List[Dict[str, Any]]:
    """
    Group sentences into overlapping windows, respecting scene boundaries.

    :return: List of ``{ window_id, sentences, text, start_sid, end_sid }``.
    """
    # Convert scene boundaries to sentence IDs
    scene_sids = set()
    if scene_boundary_offsets:
        for offset in scene_boundary_offsets:
            scene_sids.add(_char_offset_to_sent_id(offset, sentences))

    i = 0
    window_id = 0
    windows: List[Dict[str, Any]] = []
    while i < len(sentences):
        # Collect sentences for this window
        window_sents: List[Dict] = []
        token_count = 0

        j = i
        while j < len(sentences):
            # Get current sentence and its tokens
            sent = sentences[j]
            sent_tokens = len(sent.get("tokens", sent["text"].split()))
            
            # If adding this sentence would exceed the window size AND we already have sentences
            if token_count + sent_tokens > window_tokens and len(window_sents) > 0:
                break
            
            # Add sentence tokens to the accumulator for the current window
            token_count += sent_tokens
            window_sents.append(sent)

            # Check if next sentence is a scene boundary — stop window here
            if (j + 1) < len(sentences) and sentences[j + 1]["sid"] in scene_sids:
                j += 1
                break

            j += 1

        if window_sents:
            start_off = int(window_sents[0]["start_char"])
            end_off = int(window_sents[-1]["end_char"])
            window_text = text[start_off:end_off]
            windows.append({
                "window_id": window_id,
                "sentences": window_sents,
                "text": window_text,
                "start_sid": window_sents[0]["sid"],
                "end_sid": window_sents[-1]["sid"],
                "tokens": [t for s in window_sents for t in s.get("tokens", [])]
            })
            window_id += 1

        # # --- DETERMINE NEXT WINDOW START (~ OVERLAP LOGIC) ---
        # # If we hit a scene boundary (e.g. Chapter end), don't overlap across it.
        # # Start the next window immediately at the first sentence of the new scene.
        # if scene_sids and any(s in scene_sids for s in range(i, j)):
        #     next_start = j
        # else:
        #     # Otherwise, create an overlap by stepping backward from the end 
        #     # of the current window (`j`) until we accumulate `overlap_tokens`.
        #     remaining_tokens = 0
        #     next_start = j
        #     for k in range(len(window_sents) - 1, -1, -1):
        #         remaining_tokens += len(window_sents[k].get("tokens", window_sents[k]["text"].split()))
        #         if remaining_tokens >= overlap_tokens:
        #             # Point the start of the next window (`next_start`) to the ID 
        #             # of the sentence where we reached the overlap threshold.
        #             next_start = window_sents[k]["sid"]
        #             break

        # # Make sure that we always advance by at least one sentence 
        # # to avoid getting stuck in an infinite loop.
        # if next_start <= i:
        #     next_start = i + 1  # avoid infinite loop

        # # Map sentence ID back to index
        # i = next_start

                # --- DETERMINE NEXT WINDOW START (~ OVERLAP LOGIC) ---
        # If the window stopped explicitly because it hit a scene boundary exactly at 'j'
        if scene_sids and j < len(sentences) and sentences[j]["sid"] in scene_sids:
            next_start = j
        else:
            # Overlap: step backward through the CURRENT window until we accumulate `overlap_tokens`
            remaining_tokens = 0
            next_start = j
            for k in range(len(window_sents) - 1, -1, -1):
                remaining_tokens += len(window_sents[k].get("tokens", window_sents[k]["text"].split()))
                if remaining_tokens >= overlap_tokens:
                    # 'i + k' corresponds exactly to the LIST INDEX of 'window_sents[k]'
                    next_start = i + k
                    break

        if next_start <= i:
            next_start = i + 1  # avoid infinite loop

        # Map list index back to loop variable
        i = next_start
        
        # Stop entirely if we've successfully reached the end of the text
        if j == len(sentences):
            break


    return windows


# ============================================================================
# WINDOW HEADER
# ============================================================================

def build_window_header(
    active_chars: set[int],
    alias_dict: dict,
    names_df: pd.DataFrame,
) -> str:
    """
    Construct a scene context header for a coref window.

    :return: Header string like ``[SCENE CONTEXT: Active characters: ...]``.
    """
    if not active_chars:
        return ""

    id_to_name = dict(zip(names_df["id"], names_df["fullname"]))
    id_to_gender = dict(zip(names_df["id"], names_df["gender"]))

    parts = []
    for cid in sorted(active_chars):
        name = id_to_name.get(cid, f"char_{cid}")
        gender = id_to_gender.get(cid, "u")
        parts.append(f"{name} ({gender})")

    return f"[SCENE CONTEXT: Active characters: {', '.join(parts)}]"


# ============================================================================
# ACTIVE CHARACTER TRACKING
# ============================================================================

def track_active_characters(
    resolved_spans: list[dict],
    current_sid: int,
    decay_window: int = 30,
) -> set[int]:
    """
    Return the set of canonical character IDs mentioned within
    the last ``decay_window`` sentences.
    """
    active = set()
    for span in resolved_spans:
        if span.get("canonical_id") is not None:
            span_sid = span.get("sid", 0)
            if current_sid - decay_window <= span_sid <= current_sid:
                active.add(span["canonical_id"])
    return active


# ============================================================================
# COREFERENCE RESOLUTION
# ============================================================================

def _load_coref_model(model_name: str = "maverick") -> Any:
    """Load the specified coref model."""

    if model_name == "maverick":
        import torch
        from maverick import Maverick

        # Fix for PyTorch 2.6+ weights_only=True security restriction.
        # Maverick checkpoints contain complex Python objects (OmegaConf, typing.Any, etc).
        # Since we trust the model, we monkeypatch torch.load to bypass the restriction permanently.
        original_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load

        model = Maverick(device="cpu", hf_name_or_path="sapienzanlp/maverick-mes-litbank")
        # The checkpoint has mixed precision: DeBERTa backbone is float16
        # but classifier heads are float32. Convert everything to half
        # (float16) to match the backbone's native precision — the model
        # was trained/calibrated in float16 and produces 0 clusters if
        # run in float32.
        model.model = model.model.half()
        torch.load = original_torch_load

        # Monkey-patch predict to cast mask tensors to the model's native
        # dtype (float16).  The library creates attention_mask (int64) and
        # eos_mask (float64) without casting, causing dtype mismatches.
        from maverick.common.util import original_token_offsets
        import types

        @torch.no_grad()
        def _patched_predict(self, sample, singletons=False, add_gold_clusters=None,
                            predefined_mentions=None, speakers=None):
            tokens, eos_indices, speakers, char_offsets = self.preprocess(sample, speakers)
            tokenized = self.tokenize(tokens, eos_indices, speakers, predefined_mentions, add_gold_clusters)

            _dtype = next(self.model.parameters()).dtype

            output = self.model(
                stage="test",
                input_ids=torch.tensor(tokenized["input_ids"]).unsqueeze(0).to(self.device),
                attention_mask=torch.tensor(tokenized["attention_mask"]).unsqueeze(0).to(self.device).to(_dtype),
                eos_mask=torch.tensor(tokenized["eos_mask"]).unsqueeze(0).to(self.device).to(_dtype),
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
                    )
                    .unsqueeze(0)
                    .to(self.device)
                ),
            )

            clusters_predicted = original_token_offsets(
                clusters=output["pred_dict"]["clusters"],
                subtoken_map=tokenized["subtoken_map"],
                new_token_map=tokenized["new_token_map"],
            )
            result = {}
            result["tokens"] = tokens
            result["clusters_token_offsets"] = clusters_predicted
            result["clusters_char_offsets"] = None
            result["clusters_token_text"] = [
                [" ".join(tokens[span[0] : span[1] + 1]) for span in cluster]
                for cluster in clusters_predicted
            ]
            result["clusters_char_text"] = None
            if char_offsets is not None:
                result["clusters_char_offsets"] = [
                    [(char_offsets[span[0]][0], char_offsets[span[1]][1]) for span in cluster]
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


def run_coref(window_input: Any, model: Any, model_name: str = "maverick") -> List:
    """
    Run the coref model on a single window.
    For Maverick: takes list[list[str]] (sentences format) -> returns list of token offset clusters.
    For fastcoref: takes str -> returns list of char offset clusters.
    """
    if model_name == "maverick":
        preds = model.predict(window_input)
        return preds.get("clusters_token_offsets", [])
    
    elif model_name == "fastcoref":
        preds = model.predict(texts=[window_input])
        clusters = []
        for cluster in preds[0].get_clusters(as_strings=False):
            c = []
            for start, end in cluster:
                c.append({
                    "text": window_input[start:end],
                    "start_char": start,
                    "end_char": end,
                })
            clusters.append(c)
        return clusters
    else:
        return []


# ============================================================================
# CLUSTER → CHARACTER MAPPING
# ============================================================================
def map_clusters_to_characters(
    clusters: List[Any],
    alias_dict: Dict[Any],
    names_df: pd.DataFrame,
    window_start_char: int = 0,
    tokens: List[Dict[Any]] = None,
    full_text: str = "",
    is_token_indexed: bool = False,
) -> Tuple[List[Dict], List[List[Dict]], int]:
    """
    Map coref clusters to canonical characters using the alias dict.

    :return: ``(resolved_spans, unknown_clusters)``
    """
    variant_to_ids, all_variants, id_to_gender, all_name_tokens = build_variant_index(names_df)
    resolved_spans: List[Dict] = []
    unknown_clusters: List[List[Dict]] = []
    num_dropped_clusters = 0

    for cluster in clusters:
        # Snap and globalize all bounds to underlying tokens
        snapped_cluster = []
        for span in cluster:

            if is_token_indexed:
                start_tok, end_tok = span
                final_start = int(tokens[start_tok]["byte_onset"])
                final_end = int(tokens[end_tok]["byte_offset"])
                final_text = full_text[final_start:final_end]

                # print(final_text)
            else:
                global_start = window_start_char + span["start_char"]
                global_end = window_start_char + span["end_char"]
                if tokens and full_text:
                    final_start, final_end, final_text = snap_span(global_start, global_end, tokens, full_text)
                else:
                    final_start, final_end, final_text = global_start, global_end, span["text"]
            
            snapped_cluster.append({
                "start_char": final_start,
                "end_char": final_end,
                "text": final_text
            })
            
        cluster = snapped_cluster
        
        # 1. Tally available explicit anchors in the cluster
        cluster_anchors = []

        for span in cluster:
            span_text = span["text"]

            # Try direct text lookup first (named mentions, nominals)
            if span_text in alias_dict:
                cluster_anchors.append({
                    "span": span,
                    "canonical_id": alias_dict[span_text]["canonical_id"],
                    "fullname": alias_dict[span_text]["fullname"],
                    "confidence": alias_dict[span_text]["score"]
                })
                continue
                
            # Try span-indexed pronoun lookup (resolved in a previous pass)
            pronoun_key = f"__pronoun__{span['start_char']}_{span['end_char']}"
            if pronoun_key in alias_dict:
                cluster_anchors.append({
                    "span": span,
                    "canonical_id": alias_dict[pronoun_key]["canonical_id"],
                    "fullname": alias_dict[pronoun_key]["fullname"],
                    "confidence": alias_dict[pronoun_key]["score"]
                })

        # Try fuzzy match w/ the longest available span of text 
        # in the current cluster if no anchor found
        if not cluster_anchors:
            longest_span = max(cluster, key=lambda s: len(s["text"]))

            # print(f'{longest_span["text"]=}')
            cid, fname, score, _ = match_name(
                longest_span["text"], "u",
                variant_to_ids, all_variants, id_to_gender,
                names_df, all_name_tokens,
                threshold=FUZZY_AUTO_ACCEPT,
            )
            
            # Fuzzy character search successful
            if cid is not None:
                cluster_anchors.append({
                    "span": longest_span,
                    "canonical_id": cid,
                    "fullname": fname,
                    "confidence": score
                })

        # 2. Assign Clusters Check conflicts
        if not cluster_anchors:
            # Clusters cannot be resolved
            shifted_cluster = []
            for span in cluster:
                shifted_cluster.append({
                    "text": span["text"],
                    "start_char": span["start_char"],
                    "end_char": span["end_char"],
                })
            unknown_clusters.append(shifted_cluster)
            continue

        from collections import Counter
        id_counts = Counter(a["canonical_id"] for a in cluster_anchors)
        unique_ids = set(id_counts.keys())
        majority_id, maj_count = id_counts.most_common(1)[0]
        
        is_conflicting = len(unique_ids) > 1
        dropped_spans = []
        cluster_dropped_partially = False

        for span in cluster:
            anchor_match = next((a for a in cluster_anchors if a["span"] == span), None)
            
            if anchor_match:
                # Explicit match - preserve exactly as anchored
                resolved_spans.append({
                    "text": span["text"],
                    "start_char": span["start_char"],
                    "end_char": span["end_char"],
                    "canonical_id": int(anchor_match["canonical_id"]),
                    "fullname": anchor_match["fullname"],
                    "confidence": round(anchor_match["confidence"], 2),
                })
            else:
                span_lower = span["text"].lower().strip()
                if is_conflicting:
                    conflicting_genders = {id_to_gender.get(cid, 'u') for cid in unique_ids}
                    
                    if span_lower in MALE_PRONOUNS or span_lower in FEMALE_PRONOUNS:
                        p_gender = 'm' if span_lower in MALE_PRONOUNS else 'f'
                        matching_cids = [cid for cid in unique_ids if id_to_gender.get(cid, 'u') == p_gender]
                        
                        if len(matching_cids) == 1:
                            assigned_id = matching_cids[0]
                            info = next(a for a in cluster_anchors if a["canonical_id"] == assigned_id)
                            resolved_spans.append({
                                "text": span["text"],
                                "start_char": span["start_char"],
                                "end_char": span["end_char"],
                                "canonical_id": int(assigned_id),
                                "fullname": info["fullname"],
                                "confidence": round(info["confidence"], 2),
                            })
                        else:
                            # Conflicting genders on pronoun level. Drop.
                            dropped_spans.append(span)
                            cluster_dropped_partially = True
                    else:
                        # Nominal/unrecognized inside conflicting cluster. Drop.
                        dropped_spans.append(span)
                        cluster_dropped_partially = True
                else:
                    # Clean assignment to majority ID
                    info = next(a for a in cluster_anchors if a["canonical_id"] == majority_id)
                    resolved_spans.append({
                        "text": span["text"],
                        "start_char": span["start_char"],
                        "end_char": span["end_char"],
                        "canonical_id": int(majority_id),
                        "fullname": info["fullname"],
                        "confidence": round(info["confidence"], 2),
                    })
                    
        if cluster_dropped_partially:
            num_dropped_clusters += 1
            if dropped_spans:
                shifted_cluster = []
                for span in dropped_spans:
                    shifted_cluster.append({
                        "text": span["text"],
                        "start_char": span["start_char"],
                        "end_char": span["end_char"],
                    })
                unknown_clusters.append(shifted_cluster)

    return resolved_spans, unknown_clusters, num_dropped_clusters


# ============================================================================
# BACK-PROPAGATION
# ============================================================================

def back_propagate(alias_dict: dict, resolved_spans: list[dict]) -> dict:
    """
    Extend the alias dict with newly resolved nominal references.
    Pronoun entries are stored as span-indexed keys to avoid collisions.
    """
    extended = dict(alias_dict)  # shallow copy

    for span in resolved_spans:
        text = span["text"]
        if span.get("canonical_id") is None:
            continue

        text_lower = text.lower().strip()
        if text_lower in PRONOUNS:
            # Span-indexed key for pronouns
            key = f"__pronoun__{span['start_char']}_{span['end_char']}"
        else:
            key = text

        if key not in extended:
            extended[key] = {
                "canonical_id": span["canonical_id"],
                "fullname": span.get("fullname", ""),
                "score": span.get("confidence", 0.0),
            }

    return extended


# ============================================================================
# SAVE
# ============================================================================

def save_stage2(
    span_index: list[dict],
    alias_dict: dict,
    unknown_clusters: list,
    out_dir: Path,
) -> Path:
    """Write Stage 2 outputs to disk."""
    stage_dir = out_dir / "coreference"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # span_index.jsonl
    with open(stage_dir / "span_index.jsonl", "w", encoding="utf-8") as f:
        for span in span_index:
            f.write(json.dumps(span, ensure_ascii=False) + "\n")

    # alias_dict_extended.json
    with open(stage_dir / "alias_dict_extended.json", "w", encoding="utf-8") as f:
        json.dump(alias_dict, f, indent=2, ensure_ascii=False)

    # unknown_clusters.json
    with open(stage_dir / "unknown_clusters.json", "w", encoding="utf-8") as f:
        json.dump(unknown_clusters, f, indent=2, ensure_ascii=False)
    
    return stage_dir


def to_ontonotes_format_with_map(win: dict):
    """
    Convert window sentences to OntoNotes word lists, splitting at quote
    tokens and discarding them. Returns:
      - window_sents_words: list[list[str]] ready for Maverick
      - index_map: dict mapping pseudo_token_idx → original win["tokens"] idx
    """
    quotes = {'"', '\u201c', '\u201d', "''", "``", "'"}
    window_sents_words = []
    index_map = {}
    pseudo_idx = 0

    # win["tokens"] is a flat list of all tokens across all sentences
    # We need a parallel flat index into it
    orig_flat_idx = 0

    for sent in win["sentences"]:
        sent_tokens = sent.get("tokens", [])
        current_sent = []
        current_orig_indices = []  # orig_flat_idx values for each token in current_sent

        for t in sent_tokens:
            word = str(t["word"])
            if word in quotes:
                # Flush current pseudo-sentence before the quote boundary
                if current_sent:
                    window_sents_words.append(current_sent)
                    for local_i, orig_i in enumerate(current_orig_indices):
                        index_map[pseudo_idx + local_i] = orig_i
                    pseudo_idx += len(current_sent)
                    current_sent = []
                    current_orig_indices = []
                # Quote token is discarded — no entry in index_map, but
                # orig_flat_idx still advances
            else:
                current_sent.append(word)
                current_orig_indices.append(orig_flat_idx)

            orig_flat_idx += 1  # always advance, quote or not

        # Flush remaining tokens at end of sentence
        if current_sent:
            window_sents_words.append(current_sent)
            for local_i, orig_i in enumerate(current_orig_indices):
                index_map[pseudo_idx + local_i] = orig_i
            pseudo_idx += len(current_sent)

    return window_sents_words, index_map


def remap_clusters(clusters, index_map):
    """
    Remap cluster token offsets from pseudo-token space back to
    original win["tokens"] space, dropping any spans with unmapped indices.
    """
    remapped = []
    for cluster in clusters:
        clean_cluster = []
        for start, end in cluster:
            if start in index_map and end in index_map:
                clean_cluster.append((index_map[start], index_map[end]))
            # If either index is missing it was a quote token — discard span
        if clean_cluster:
            remapped.append(clean_cluster)
    return remapped


def run_coref_pass(windows: List[Dict], already_resolved: List[Dict], alias_dict, clean_names, coref_model, coref_model_name, text):
    all_resolved: list[dict] = already_resolved
    all_unknown: list[list[dict]] = []
    total_dropped_clusters = 0

    for win in tqdm(windows):
        if coref_model_name == "maverick":
            # Convert window into OntoNotes format. Remove quotation marks
            # print(*win)
            window_sents_words, idx_map = to_ontonotes_format_with_map(win)
            # print(window_sents_words)

            # Predict clusters
            clusters_token_idx = run_coref(window_sents_words, coref_model, coref_model_name)

            # Remap clusters to token IDs w/ quotation marks
            clusters_token_idx = remap_clusters(clusters_token_idx, idx_map)

            # Map clusters to characters
            resolved, unknown, dropped_count = map_clusters_to_characters(
                clusters_token_idx, alias_dict, clean_names, 
                tokens=win.get("tokens", []), full_text=text, is_token_indexed=True
            )
            total_dropped_clusters += dropped_count

        all_resolved.extend(resolved)
        all_unknown.extend(unknown)

        # Add sentence IDs to resolved spans
        existing_keys = {(s["start_char"], s["end_char"]) for s in all_resolved}
        for span in resolved:
            key = (span.get("start_char", 0), span.get("end_char", 0))
            if key not in existing_keys:
                all_resolved.append(span)
                existing_keys.add(key)

    return all_resolved, all_unknown, total_dropped_clusters


def _worker_init(model_name):
    """Called once per worker process — loads model into process-local global."""
    global _worker_model
    _worker_model = _load_coref_model(model_name)

def _process_window(args):
    win, alias_dict, clean_names, coref_model_name, text = args
    window_sents_words, idx_map = to_ontonotes_format_with_map(win)
    # window_sents_words = [[str(t["word"]) for t in s.get("tokens", [])] for s in win["sentences"]]
    clusters_token_idx = run_coref(window_sents_words, _worker_model, coref_model_name)
    clusters_token_idx = remap_clusters(clusters_token_idx, idx_map)
    resolved, unknown, dropped_count = map_clusters_to_characters(
                clusters_token_idx, alias_dict, clean_names, 
                tokens=win.get("tokens", []), full_text=text, is_token_indexed=True
            )
    return resolved, unknown, dropped_count

def run_coref_pass_parallel(windows, already_resolved, alias_dict, clean_names,
                             coref_model, coref_model_name, text, n_workers=4):
    args = [(win, alias_dict, clean_names, coref_model_name, text) for win in windows]

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(coref_model_name,)
    ) as pool:
        results = pool.map(_process_window, args)

    all_resolved = list(already_resolved)
    all_unknown = []
    total_dropped_clusters = 0
    seen = {(s["start_char"], s["end_char"]) for s in all_resolved}
    for resolved, unknown, dropped in results:
        total_dropped_clusters += dropped
        for span in resolved:
            key = (span.get("start_char", 0), span.get("end_char", 0))
            if key not in seen:
                all_resolved.append(span)
                seen.add(key)
        all_unknown.extend(unknown)

    return all_resolved, all_unknown, total_dropped_clusters

# ============================================================================
# MAIN
# ============================================================================

def main(
    text_path: Path = RAW_TEXT,
    char_res_dir: Path = CHAR_RES_OUT,
    names_csv: Path = CLEAN_NAMES,
    out_dir: Path = COREF_OUT,
    tokens_path: Path = None,
    coref_model_name: str = "maverick",
    refine: bool = False,
    verbose: bool = False,
    threaded: bool = False
) -> Tuple[List[Dict], Dict]:
    """Run Stage 2: coreference resolution → span index."""

    from warnings import simplefilter
    simplefilter("ignore")

    print_headers("STAGE 2 — COREFERENCE RESOLUTION", "=", prefix="\n")

    # --------------------- LOAD INPUTS ---------------------
    print_information("Loading inputs...", 1, "\n")
    text = load_text(text_path) # text_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    alias_dict = load_alias_dict(char_res_dir / "alias_dict.json")
    clean_names = pd.read_csv(names_csv)
    print_information(f"Alias dict has {len(alias_dict)} entries", prefix="    ")

    # ------------------ SEGMENT SENTENCES ------------------
    print_information("Segmenting sentences from tokens file...", 3, "\n")
    from .character_resolution import segment_sentences
    sentences = segment_sentences(text, tokens_path)
    print_information(f"Found {len(sentences)} sentences", prefix="    ")

    # --------------- DETECT SCENE BOUNDARIES ---------------
    print_information("Detecting scene boundaries...", 4, "\n")
    scene_boundaries = detect_scene_boundaries(text)
    print_information(f"Found {len(scene_boundaries)} scene boundaries", prefix="    ")

    # -------------------- BUILD WINDOWS --------------------
    print_information("Building coref windows...", 5, "\n")
    windows = build_windows(sentences, text, scene_boundary_offsets=scene_boundaries)
    print_information(f"Built {len(windows)} windows", prefix="    ")

    # ------------------ LOAD COREF MODEL -------------------
    print_information(f"Loading coref model '{coref_model_name}'...", 6, "\n")
    with suppress_stdout():
        coref_model = _load_coref_model(coref_model_name)
    print_information("Model loaded", prefix="    ")

    # ----------------- RUN COREF PER WINDOW ----------------
    print_information("Running coreference resolution...", 7, "\n")
    params = {"windows": windows, "already_resolved": [], "alias_dict": alias_dict, "clean_names": clean_names, "coref_model": coref_model, "coref_model_name": coref_model_name, "text": text}
    if not threaded:
        all_resolved, all_unknown, dropped = run_coref_pass(windows, [], alias_dict, clean_names, coref_model, coref_model_name, text)
    else:
        all_resolved, all_unknown, dropped = run_coref_pass_parallel(**params)
    print_information(f"Resolved {len(all_resolved)} spans", prefix="    ")
    print_information(f"Unknown clusters: {len(all_unknown)}", prefix="    ")

    # ---------------- BACK-PROGATE RESULTS -----------------
    print_information("Back-propagating to alias dictionary...", 8, "\n")
    extended_dict = back_propagate(alias_dict, all_resolved)
    new_entries = len(extended_dict) - len(alias_dict)
    print_information(f"Added {new_entries} new entries to alias dict", prefix="    ")

    # ----------------- RUN REFINEMENT PASS -----------------
    if refine:
        print_information("Running refinement pass...", 9, "\n")
        # Re-run with extended dict
        # all_resolved, all_unknown, _ = run_coref_pass_parallel(windows, [], alias_dict, clean_names, coref_model, coref_model_name, text)
        all_resolved, all_unknown, dropped_refine = run_coref_pass(windows, all_resolved, extended_dict, clean_names, coref_model, coref_model_name, text)
        
        # Back-propagate refinement pass results
        extended_dict = back_propagate(extended_dict, all_resolved)
        
        # Add dropped spans due to inability to resolve ambiguious spans in cluster
        dropped += dropped_refine
        print_information(f"After refinement: {len(all_resolved)} total resolved spans", prefix="    ")
        print_information(f"After refinement: {len(all_unknown)} spans remain unresolved", prefix="    ")

    # ----------------- PRINT VERBOSE STATS -----------------
    if verbose and all_resolved:
        print_verbose_stats(all_resolved, alias_dict, extended_dict, clean_names, all_unknown, dropped)

    # --------------------- SAVE RESULTS --------------------
    print_information("Saving Coreference outputs...", symb=(10 if refine else 9), prefix="\n")
    out_dir = save_stage2(all_resolved, extended_dict, all_unknown, out_dir)
    print_information(f"Saved to → {out_dir}", "✓", col="GREEN")

    # --------------- VISUALIZE COREF RESULTS ---------------
    from .viz_resolution import main as viz_main
    print_information("Generating Coreference visualization...", symb=(11 if refine else 10), prefix="\n")
    viz_main(text=text_path, 
             spans_path=out_dir / "span_index.jsonl", 
             unknown=out_dir / "unknown_clusters.json", 
             out=out_dir / "coreference_visualization.html")
    print_information(f"Saved to → {out_dir / 'coref_visualization.html'}", "✓", col="GREEN")
    
    # TODO: Probably remove this
    return out_dir


if __name__ == "__main__":
    import argparse
    from .config import RAW_TEXT, CLEAN_NAMES

    parser = argparse.ArgumentParser(description="Stage 2: Coreference Resolution")
    parser.add_argument("--text-path", type=Path, default=RAW_TEXT)
    parser.add_argument("--char-res-dir", type=Path, default=CHAR_RES_OUT)
    parser.add_argument("--names-csv", type=Path, default=CLEAN_NAMES)
    parser.add_argument("--out-dir", type=Path, default=COREF_OUT)
    parser.add_argument("--tokens-path", type=Path, default=TOKENS)
    parser.add_argument("--coref-model", type=str, default="maverick")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-t", "--threaded", action="store_true")

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
        threaded=args.threaded
    )
