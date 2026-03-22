"""
Stage 2 — Coreference Resolution

Resolve pronouns, possessives, and nominal references to canonical
character IDs using a neural coref model (Maverick or FastCoref),
anchored by the Stage 1 alias dictionary.
"""
from __future__ import annotations

import json
import re
import spacy
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import Any, Optional

from .config import (
    CLEAN_NAMES, OUT_DIR, TOKENS,
    COREF_WINDOW_TOKENS, COREF_OVERLAP_TOKENS,
    FUZZY_AUTO_ACCEPT, NER_OUT, RAW_TEXT, PRONOUNS
)
from .utils import (
    build_variant_index, match_name,
    print_headers, print_information,
)


# ============================================================================
# LOADING
# ============================================================================

def load_alias_dict(path: Path) -> dict:
    """Load alias_dict.json from Stage 1 output."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sentences(ner_dir: Path) -> list[dict]:
    """
    Reconstruct the sentence list from ner_spans.jsonl.
    Each sentence gets its text from the span records.
    """
    spans = []
    with open(ner_dir / "ner_spans.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            spans.append(json.loads(line))
    return spans


# ============================================================================
# SCENE BOUNDARIES
# ============================================================================

def detect_scene_boundaries(text: str) -> list[int]:
    """
    Detect hard scene boundaries: Markdown chapter/section markers (# and ##),
    'Chapter' text markers, '***', '---'.

    :return: List of character offsets where new scenes begin.
    """
    boundaries: list[int] = []

    # Markdown chapter markers (# Chapter... or ## Section...)
    for m in re.finditer(r'(?m)^(#{1,2})\s+.*$', text): # \s*
        boundaries.append(m.start())

    # Chapter markers
    for m in re.finditer(r'\bChapter\s+\d+', text, re.IGNORECASE):
        boundaries.append(m.start())

    # Epilogue / Manuscript markers
    for m in re.finditer(r'\b(Epilogue|MANUSCRIPT)\b', text):
        boundaries.append(m.start())

    # Separator lines
    for m in re.finditer(r'\n\s*(\*\*\*|---)\s*\n', text):
        boundaries.append(m.start())

    return sorted(set(boundaries))


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
    sentences: list[dict],
    text: str,
    window_tokens: int = COREF_WINDOW_TOKENS,
    overlap_tokens: int = COREF_OVERLAP_TOKENS,
    scene_boundary_offsets: list[int] | None = None,
) -> list[dict]:
    """
    Group sentences into overlapping windows, respecting scene boundaries.

    :return: List of ``{ window_id, sentences, text, start_sid, end_sid }``.
    """
    # Convert scene boundaries to sentence IDs
    scene_sids = set()
    if scene_boundary_offsets:
        i = 0
        for offset in scene_boundary_offsets:
            scene_sids.add(_char_offset_to_sent_id(offset, sentences))
            if i == 2:
                break
            i += 1

    print(scene_sids)
    windows: list[dict] = []
    window_id = 0
    i = 0

    while i < len(sentences):
        # Collect sentences for this window
        window_sents: list[dict] = []
        token_count = 0

        j = i
        while j < len(sentences):
            sent = sentences[j]
            sent_tokens = len(sent["text"].split())
            
            # If adding this sentence would exceed the window size AND we already have sentences
            if token_count + sent_tokens > window_tokens and len(window_sents) > 0:
                break
                
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

        # Advance with overlap
        overlap_count = 0
        advance = i
        for k in range(len(window_sents) - 1, -1, -1):
            overlap_count += len(window_sents[k]["text"].split())
            if overlap_count >= overlap_tokens:
                advance = window_sents[k]["sid"]
                break

        # If we hit a scene boundary, don't overlap across it
        next_start = max(j - (len(window_sents) - (advance - i)), j)
        if scene_sids and any(s in scene_sids for s in range(i, j)):
            next_start = j
        else:
            # Overlap: go back by overlap_tokens worth of sentences
            remaining_tokens = 0
            next_start = j
            for k in range(len(window_sents) - 1, -1, -1):
                remaining_tokens += len(window_sents[k]["text"].split())
                if remaining_tokens >= overlap_tokens:
                    next_start = window_sents[k]["sid"]
                    break

        if next_start <= i:
            next_start = i + 1  # avoid infinite loop

        # Map sentence ID back to index
        i = next_start

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


def run_coref(window_input: Any, model: Any, model_name: str = "maverick") -> list:
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

def snap_span(start: int, end: int, tokens: list[dict], full_text: str) -> tuple[int, int, str]:
    matched = [t for t in tokens if t["byte_offset"] > start and t["byte_onset"] < end]
    if matched:
        s = int(matched[0]["byte_onset"])
        e = int(matched[-1]["byte_offset"])
        return s, e, full_text[s:e]
    return start, end, full_text[start:end]

def map_clusters_to_characters(
    clusters: list,
    alias_dict: dict,
    names_df: pd.DataFrame,
    window_start_char: int = 0,
    tokens: list[dict] = None,
    full_text: str = "",
    is_token_indexed: bool = False,
) -> tuple[list[dict], list[list[dict]]]:
    """
    Map coref clusters to canonical characters using the alias dict.

    :return: ``(resolved_spans, unknown_clusters)``
    """
    variant_to_ids, all_variants, id_to_gender, all_name_tokens = build_variant_index(names_df)
    resolved_spans: list[dict] = []
    unknown_clusters: list[list[dict]] = []

    for cluster in clusters:
        # Snap and globalize all bounds to underlying tokens
        snapped_cluster = []
        for span in cluster:
            if is_token_indexed:
                start_tok, end_tok = span
                final_start = int(tokens[start_tok]["byte_onset"])
                final_end = int(tokens[end_tok]["byte_offset"])
                final_text = full_text[final_start:final_end]
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
        
        # Try to find an anchor in the alias dict
        canonical_id = None
        fullname = None
        confidence = 0.0

        for span in cluster:
            span_text = span["text"]
            if span_text in alias_dict:
                canonical_id = alias_dict[span_text]["canonical_id"]
                fullname = alias_dict[span_text]["fullname"]
                confidence = alias_dict[span_text]["score"]
                break

        # Secondary fuzzy match on longest span if no anchor found
        if canonical_id is None:
            longest_span = max(cluster, key=lambda s: len(s["text"]))
            cid, fname, score, _ = match_name(
                longest_span["text"], "u",
                variant_to_ids, all_variants, id_to_gender,
                names_df, all_name_tokens,
                threshold=FUZZY_AUTO_ACCEPT,
            )
            if cid is not None:
                canonical_id = cid
                fullname = fname
                confidence = score

        if canonical_id is not None:
            for span in cluster:
                resolved_spans.append({
                    "text": span["text"],
                    "start_char": span["start_char"],
                    "end_char": span["end_char"],
                    "canonical_id": int(canonical_id),
                    "fullname": fullname,
                    "confidence": round(confidence, 2),
                })
        else:
            shifted_cluster = []
            for span in cluster:
                shifted_cluster.append({
                    "text": span["text"],
                    "start_char": span["start_char"],
                    "end_char": span["end_char"],
                })
            unknown_clusters.append(shifted_cluster)

    return resolved_spans, unknown_clusters


# ============================================================================
# BACK-PROPAGATION
# ============================================================================

def back_propagate(alias_dict: dict, resolved_spans: list[dict]) -> dict:
    """
    Extend the alias dict with newly resolved **nominal references** discovered
    by coreference resolution (e.g. "the soldier", "the judge").

    Pronouns are intentionally excluded. Storing "he", "she", "they" etc. as
    string keys in the alias dict is unsound because the same pronoun string
    can refer to any number of different characters across the book — their
    mapping is inherently span-indexed, not text-indexed.  Pronoun resolutions
    are already captured in the ``span_index.jsonl`` output and should be
    looked up there by character offset in Stages 3/4, not via the alias dict.

    Only multi-token or capitalized nominal spans (e.g. "the old soldier",
    "her host") that are NOT bare pronouns and NOT already present in the
    alias dict are added.
    """
    extended = dict(alias_dict)  # shallow copy

    for span in resolved_spans:
        text = span["text"]
        cid  = span.get("canonical_id")

        if cid is None:
            continue

        text_lower = text.lower().strip()

        # Skip pronouns entirely — they must never pollute the alias dict.
        if text_lower in PRONOUNS:
            continue

        # Skip surface forms already in the dict (Stage 1 entries take
        # precedence; they were anchored with higher confidence).
        if text in extended:
            continue

        # Only add spans that look like nominal references: at least two
        # tokens (articles + noun, possessives + noun, etc.) or a single
        # token that is capitalised (i.e. a name-like form NER missed).
        tokens = text.split()
        is_nominal = len(tokens) >= 2 or (len(tokens) == 1 and tokens[0][0].isupper())

        if is_nominal:
            extended[text] = {
                "canonical_id": span["canonical_id"],
                "fullname":     span.get("fullname", ""),
                "score":        span.get("confidence", 0.0),
                "source":       "coref_backprop",
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
) -> None:
    """Write Stage 2 outputs to disk."""
    stage_dir = out_dir / "stage2"
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


def run_coref_pass(windows: List[Dict], already_resolved: List[Dict], alias_dict, clean_names, coref_model, coref_model_name, text):
    all_resolved: list[dict] = already_resolved
    all_unknown: list[list[dict]] = []

    for win in tqdm(windows):
        # Build header for active characters
        active = track_active_characters(all_resolved, win["start_sid"])
        header = build_window_header(active, alias_dict, clean_names)
        window_text = (header + " " + win["text"]) if header else win["text"]

        if coref_model_name == "maverick":
            # Pass sentences as list of word lists (OntoNotes format)
            window_sents_words = []
            for sent in win["sentences"]:
                window_sents_words.append([str(t["word"]) for t in sent.get("tokens", [])])
                
            clusters_token_idx = run_coref(window_sents_words, coref_model, coref_model_name)
            
            resolved, unknown = map_clusters_to_characters(
                clusters_token_idx, alias_dict, clean_names, 
                tokens=win.get("tokens", []), full_text=text, is_token_indexed=True
            )
        else:
            # Run coref
            clusters = run_coref(window_text, coref_model, coref_model_name)

            # TODO: REVIEW from here -->
            # If a header was prepended, the coref model's character offsets
            # are relative to the header-prefixed window_text.  Shift them
            # back so they are relative to the original win["text"].
            header_len = (len(header) + 1) if header else 0  # +1 for the separating space
            if header_len > 0:
                cleaned_clusters = []
                for cluster in clusters:
                    new_cluster = []
                    for span in cluster:
                        s_start = span["start_char"] - header_len
                        s_end = span["end_char"] - header_len
                        if s_start >= 0:
                            new_cluster.append({
                                "text": win["text"][s_start:s_end],
                                "start_char": s_start,
                                "end_char": s_end
                            })
                    if new_cluster:
                        cleaned_clusters.append(new_cluster)
                clusters = cleaned_clusters
            # TODO: to here <--

            # Map clusters to characters
            # Adjust start_char offset: use the first sentence's start_char
            win_start_char = win["sentences"][0]["start_char"] if win["sentences"] else 0
            resolved, unknown = map_clusters_to_characters(
                clusters, alias_dict, names_df, win_start_char, win.get("tokens"), text, is_token_indexed=False
            )

        # all_resolved.extend(resolved)
        all_unknown.extend(unknown)

        # Add sentence IDs to resolved spans
        existing_keys = {(s["start_char"], s["end_char"]) for s in all_resolved}
        for span in resolved:
            key = (span.get("start_char", 0), span.get("end_char", 0))
            if key not in existing_keys:
                all_resolved.append(span)
                existing_keys.add(key)
            
        # for span in resolved:
        #     for sent in win["sentences"]:
        #         if sent["start_char"] <= span.get("start_char", 0) < sent["end_char"]:
        #             span["sid"] = sent["sid"]
        #             break

    return all_resolved, all_unknown


def print_statistics(all_resolved, alias_dict, extended_dict, clean_names, all_unknown):
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
        print(f"\n    Confidence Distribution:")
        print(f"      Min: {min(confidences):.1f}  Max: {max(confidences):.1f}  "
                f"Mean: {sum(confidences)/len(confidences):.1f}")

    # Alias dict growth
    print(f"\n    Alias Dictionary Growth:")
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

# ============================================================================
# MAIN
# ============================================================================

def main(
    text_path: Path = RAW_TEXT,
    ner_dir: Path = NER_OUT,
    names_csv: Path = CLEAN_NAMES,
    out_dir: Path = OUT_DIR,
    coref_model_name: str = "maverick",
    refine: bool = False,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Run Stage 2: coreference resolution → span index."""

    print_headers("STAGE 2 — COREFERENCE RESOLUTION", "=", prefix="\n")

    # Load inputs
    print_information("Loading inputs...", 1, "\n")
    text = text_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    alias_dict = load_alias_dict(ner_dir / "alias_dict.json")
    clean_names = pd.read_csv(names_csv)
    print_information(f"Alias dict has {len(alias_dict)} entries", prefix="    ")

    # Stage 0 tokenisation takes care of sentence segmentation
    # so we don't need to load spaCy here anymore.
    # Segment sentences
    print_information("Segmenting sentences from tokens file...", 3, "\n")
    from .stage1_character_resolution import segment_sentences
    sentences = segment_sentences(text, TOKENS)
    print_information(f"Found {len(sentences)} sentences", prefix="    ")

    # Detect scene boundaries
    print_information("Detecting scene boundaries...", 4, "\n")
    scene_boundaries = detect_scene_boundaries(text)
    print(scene_boundaries)
    print_information(f"Found {len(scene_boundaries)} scene boundaries", prefix="    ")

    # Build windows
    print_information("Building coref windows...", 5, "\n")
    windows = build_windows(sentences, text,
                            scene_boundary_offsets=scene_boundaries)
    print_information(f"Built {len(windows)} windows", prefix="    ")
    # return

    # Load coref model
    print_information(f"Loading coref model '{coref_model_name}'...", 6, "\n")
    coref_model = _load_coref_model(coref_model_name)
    print_information("Model loaded", prefix="    ")

    # Process each window
    print_information("Running coreference resolution...", 7, "\n")
    all_resolved, all_unknown = run_coref_pass(windows, [], alias_dict, clean_names, coref_model, coref_model_name, text)

    print_information(f"Resolved {len(all_resolved)} spans", prefix="    ")
    print_information(f"Unknown clusters: {len(all_unknown)}", prefix="    ")

    # Back-propagate to extend alias dict
    print_information("Back-propagating to alias dictionary...", 8, "\n")
    extended_dict = back_propagate(alias_dict, all_resolved)
    new_entries = len(extended_dict) - len(alias_dict)
    print_information(f"Added {new_entries} new entries to alias dict", prefix="    ")

    # Optional refinement pass
    if refine:
        print_information("Running refinement pass...", 9, "\n")
        # Re-run with extended dict
        all_resolved, all_unknown = run_coref_pass(windows, all_resolved, extended_dict, clean_names, coref_model, coref_model_name, text)
        # all_resolved_2: list[dict] = []
        # for win in windows:
        #     if coref_model_name == "maverick":
        #         window_sents_words = []
        #         for sent in win["sentences"]:
        #             window_sents_words.append([str(t["word"]) for t in sent.get("tokens", [])])
                    
        #         clusters_token_idx = run_coref(window_sents_words, coref_model, coref_model_name)
                
        #         resolved, unknown = map_clusters_to_characters(
        #             clusters_token_idx, extended_dict, names_df, 
        #             tokens=win.get("tokens", []), full_text=text, is_token_indexed=True
        #         )
        #     else:
        #         active = track_active_characters(all_resolved, win["start_sid"])
        #         header = build_window_header(active, extended_dict, names_df)
        #         window_text = (header + " " + win["text"]) if header else win["text"]

        #         clusters = run_coref(window_text, coref_model, coref_model_name)
                
        #         header_len = (len(header) + 1) if header else 0
        #         if header_len > 0:
        #             cleaned_clusters = []
        #             for cluster in clusters:
        #                 new_cluster = []
        #                 for span in cluster:
        #                     s_start = span["start_char"] - header_len
        #                     s_end = span["end_char"] - header_len
        #                     if s_start >= 0:
        #                         new_cluster.append({
        #                             "text": win["text"][s_start:s_end],
        #                             "start_char": s_start,
        #                             "end_char": s_end
        #                         })
        #                 if new_cluster:
        #                     cleaned_clusters.append(new_cluster)
        #             clusters = cleaned_clusters

        #         win_start_char = win["sentences"][0]["start_char"] if win["sentences"] else 0
        #         resolved, unknown = map_clusters_to_characters(
        #             clusters, extended_dict, names_df, win_start_char, win.get("tokens"), text, is_token_indexed=False
        #         )
                
        #     all_resolved_2.extend(resolved)

        # # Merge new resolutions
        # existing_keys = {(s["start_char"], s["end_char"]) for s in all_resolved}
        # for span in all_resolved_2:
        #     key = (span.get("start_char", 0), span.get("end_char", 0))
        #     if key not in existing_keys:
        #         all_resolved.append(span)
        #         existing_keys.add(key)

        extended_dict = back_propagate(extended_dict, all_resolved)
        print_information(f"After refinement: {len(all_resolved)} total resolved spans", prefix="    ")

    # Verbose statistics
    if verbose and all_resolved:
        print_statistics(all_resolved, alias_dict, extended_dict, clean_names, all_unknown)

    # Save
    print_information("Saving Stage 2 outputs...", 10, "\n")
    save_stage2(all_resolved, extended_dict, all_unknown, out_dir)
    print_information(f"Saved to → {out_dir / 'stage2'}", "✓", col="GREEN")

    return all_resolved, extended_dict


if __name__ == "__main__":
    import argparse
    from .config import RAW_TEXT, CLEAN_NAMES

    parser = argparse.ArgumentParser(description="Stage 2: Coreference Resolution")
    parser.add_argument("--text-path", type=Path, default=RAW_TEXT)
    parser.add_argument("--ner-dir", type=Path, required=True)
    parser.add_argument("--names-csv", type=Path, default=CLEAN_NAMES)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--coref-model", type=str, default="maverick")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    main(
        text_path=args.text_path,
        ner_dir=args.ner_dir,
        names_csv=args.names_csv,
        out_dir=args.out_dir,
        coref_model_name=args.coref_model,
        refine=args.refine,
        verbose=args.verbose,
    )
