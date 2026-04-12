"""
verify_triples.py — Interactive HTML Visualization for Triple Verification

Reads extracted triples and produces an HTML file that:
  1. Highlights agent, verb, and patient spans in the original text (green)
  2. Shows canonical names as subscript labels
  3. Displays triple graph annotations in the right margin
  4. Allows double-click editing of triple components
  5. Exports all triples (with corrections) as a downloadable TSV
"""
from __future__ import annotations

import argparse
import math
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import OUT_DIR, RAW_TEXT, TOKENS, COREF_OUT, PRONOUNS
from ..extraction.utils import load_text, load_span_index, print_information, print_headers


# ============================================================================
# DATA LOADING
# ============================================================================

def load_triples(triples_path: Path) -> pd.DataFrame:
    """Load triples.csv (TSV)."""
    import pandas as pd
    df = pd.read_csv(triples_path, sep="\t", keep_default_na=False)
    return df


def load_tokens(tokens_path: Path) -> pd.DataFrame:
    """Load the .tokens file as a DataFrame."""
    import pandas as pd
    df = pd.read_csv(tokens_path, sep="\t", keep_default_na=False)
    return df


# ============================================================================
# BYTE OFFSET LOOKUP
# ============================================================================

def build_token_index(tokens_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Build a dict mapping token_ID_within_document → row dict."""
    index = {}
    for _, row in tokens_df.iterrows():
        tid = int(row["token_ID_within_document"])
        index[tid] = {
            "byte_onset": int(row["byte_onset"]),
            "byte_offset": int(row["byte_offset"]),
            "sentence_ID": int(row["sentence_ID"]),
            "word": str(row["word"]),
        }
    return index


def build_sentence_boundaries(tokens_df: pd.DataFrame) -> Dict[int, Tuple[int, int]]:
    """
    Build a dict mapping sentence_ID → (start_byte, end_byte) using
    the byte_onset of the first token and byte_offset of the last token.
    """
    boundaries = {}
    for sid, group in tokens_df.groupby("sentence_ID"):
        sid = int(sid)
        start = int(group["byte_onset"].min())
        end = int(group["byte_offset"].max())
        boundaries[sid] = (start, end)
    return boundaries


# ============================================================================
# CHAPTER BOUNDARIES & SPLIT LOGIC
# ============================================================================

def find_chapter_boundaries(text: str) -> List[int]:
    """
    Find byte offsets of every `# Chapter` heading in the text.
    Returns a sorted list of byte positions where each chapter starts.
    The list always starts with 0 (beginning of text).
    """
    boundaries = [0]
    for m in re.finditer(r'^# .+', text, re.MULTILINE):
        pos = m.start()
        if pos not in boundaries:
            boundaries.append(pos)
    boundaries.sort()
    return boundaries


def compute_split_ranges(
    enriched_triples: List[Dict],
    chapter_bounds: List[int],
    text_len: int,
    n_splits: int,
) -> List[Tuple[int, int, List[Dict]]]:
    """
    Split triples into `n_splits` groups (sorted by verb position),
    then expand each group's range to chapter boundaries.

    Returns list of (text_start, text_end, triples_in_split).
    """
    if n_splits <= 1:
        return [(0, text_len, enriched_triples)]

    # Sort triples by verb byte position
    sorted_triples = sorted(enriched_triples, key=lambda t: t["verb_onset"])
    chunk_size = math.ceil(len(sorted_triples) / n_splits)

    ranges = []
    for i in range(n_splits):
        chunk = sorted_triples[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            continue

        first_byte = chunk[0]["sent_start"]
        last_byte = chunk[-1]["sent_end"]

        # Snap to chapter boundaries:
        # Start = largest chapter boundary <= first_byte
        chap_start = 0
        for cb in chapter_bounds:
            if cb <= first_byte:
                chap_start = cb
            else:
                break

        # End = next chapter boundary after last_byte (or end of text)
        chap_end = text_len
        for cb in chapter_bounds:
            if cb > last_byte:
                chap_end = cb
                break

        ranges.append((chap_start, chap_end, chunk))

    return ranges


# ============================================================================
# REVERSE COREFERENCE — FIND SURFACE SPANS FOR AGENT / PATIENT
# ============================================================================

def build_span_lookup_by_canonical(
    span_index: List[Dict],
) -> Dict[int, List[Dict]]:
    """
    Group spans by canonical_id for efficient lookup.
    Returns {canonical_id: [span_dict, ...]}.
    """
    lookup: Dict[int, List[Dict]] = {}
    for span in span_index:
        cid = span.get("canonical_id")
        if cid is None:
            continue
        cid = int(cid)
        if cid not in lookup:
            lookup[cid] = []
        lookup[cid].append(span)
    return lookup


def find_surface_span(
    canonical_id: int,
    verb_byte_onset: int,
    sentence_start: int,
    sentence_end: int,
    span_lookup: Dict[int, List[Dict]],
    role: str,  # "agent" or "patient"
) -> Optional[Tuple[int, int, str]]:
    """
    Find the surface-text span for a character in a given sentence.

    Strategy:
      - Filter spans for this canonical_id that overlap the sentence byte range.
      - For agent role, prefer the closest span BEFORE or AT the verb.
      - For patient role, prefer the closest span AFTER or AT the verb.
      - Fallback: closest span in the sentence regardless of position.

    Returns (start_byte, end_byte, surface_text) or None.
    """
    candidate_spans = span_lookup.get(canonical_id, [])

    # Filter to spans within sentence boundaries (with slight tolerance)
    in_sentence = []
    for sp in candidate_spans:
        sp_start = sp["start_char"]
        sp_end = sp["end_char"]
        # Span must overlap with the sentence
        if sp_start >= sentence_start and sp_end <= sentence_end:
            in_sentence.append(sp)

    if not in_sentence:
        return None

    # Deduplicate by (start_char, end_char) — span_index has duplicates
    seen = set()
    deduped = []
    for sp in in_sentence:
        key = (sp["start_char"], sp["end_char"])
        if key not in seen:
            seen.add(key)
            deduped.append(sp)
    in_sentence = deduped

    if role == "agent":
        # Prefer spans before or at the verb
        before = [sp for sp in in_sentence if sp["start_char"] <= verb_byte_onset]
        if before:
            # Closest to verb (largest start_char)
            best = max(before, key=lambda sp: sp["start_char"])
        else:
            # Fallback: closest span overall
            best = min(in_sentence, key=lambda sp: abs(sp["start_char"] - verb_byte_onset))
    else:  # patient
        # Prefer spans after or at the verb
        after = [sp for sp in in_sentence if sp["start_char"] >= verb_byte_onset]
        if after:
            # Closest to verb (smallest start_char)
            best = min(after, key=lambda sp: sp["start_char"])
        else:
            # Fallback: closest span overall
            best = min(in_sentence, key=lambda sp: abs(sp["start_char"] - verb_byte_onset))

    return (best["start_char"], best["end_char"], best["text"])


# ============================================================================
# TRIPLE ENRICHMENT — ATTACH BYTE OFFSETS TO EACH TRIPLE
# ============================================================================

def enrich_triples(
    triples_df: pd.DataFrame,
    token_index: Dict[int, Dict[str, Any]],
    sentence_bounds: Dict[int, Tuple[int, int]],
    span_lookup: Dict[int, List[Dict]],
) -> List[Dict[str, Any]]:
    """
    For each triple, resolve byte offsets for verb, agent and patient.
    Returns a list of enriched triple dicts.
    """
    enriched = []
    for row_idx, row in triples_df.iterrows():
        verb_token_id = int(row["index"])
        tok_info = token_index.get(verb_token_id)
        if tok_info is None:
            continue

        verb_onset = tok_info["byte_onset"]
        verb_offset = tok_info["byte_offset"]
        sid = tok_info["sentence_ID"]
        sent_bounds = sentence_bounds.get(sid)
        if sent_bounds is None:
            continue

        sent_start, sent_end = sent_bounds

        # Find agent surface span
        agent_cid = int(row["canonical_id_left"])
        agent_span = find_surface_span(
            agent_cid, verb_onset, sent_start, sent_end, span_lookup, "agent"
        )

        # Find patient surface span
        patient_cid = int(row["canonical_id_right"])
        patient_span = find_surface_span(
            patient_cid, verb_onset, sent_start, sent_end, span_lookup, "patient"
        )

        # Build unique key: canonical_id_left + canonical_id_right + verb_index + word
        triple_key = f"{agent_cid}_{patient_cid}_{verb_token_id}_{row['word']}"

        enriched.append({
            "row_idx": int(row_idx),
            "triple_key": triple_key,
            "sentence_id": sid,
            "sent_start": sent_start,
            "sent_end": sent_end,
            # Verb
            "verb_onset": verb_onset,
            "verb_offset": verb_offset,
            "verb_text": str(row["word"]),
            "verb_lemma": str(row["lemma"]),
            "negated": bool(row["negated"]),
            # Agent
            "agent_cid": agent_cid,
            "agent_name": str(row["name_left"]),
            "agent_span": agent_span,  # (start, end, surface_text) or None
            # Patient
            "patient_cid": patient_cid,
            "patient_name": str(row["name_right"]),
            "patient_span": patient_span,  # (start, end, surface_text) or None
        })

    return enriched


# ============================================================================
# HTML GENERATION
# ============================================================================

def _resolve_highlight_spans(triples_in_sentence: List[Dict]) -> List[Tuple[int, int, str, str, str]]:
    """
    Collect all spans to highlight in a sentence.
    Returns list of (start, end, css_class, label, triple_key).
    Deduplicates by (start, end).
    """
    spans = []
    seen = set()

    for t in triples_in_sentence:
        # Verb span
        key = (t["verb_onset"], t["verb_offset"])
        if key not in seen:
            seen.add(key)
            spans.append((t["verb_onset"], t["verb_offset"], "hl-verb", t["verb_text"], t["triple_key"]))

        # Agent span
        if t["agent_span"]:
            a_start, a_end, a_text = t["agent_span"]
            key = (a_start, a_end)
            if key not in seen:
                seen.add(key)
                spans.append((a_start, a_end, "hl-agent", t["agent_name"], t["triple_key"]))

        # Patient span
        if t["patient_span"]:
            p_start, p_end, p_text = t["patient_span"]
            key = (p_start, p_end)
            if key not in seen:
                seen.add(key)
                spans.append((p_start, p_end, "hl-patient", t["patient_name"], t["triple_key"]))

    # Sort by start position
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    return spans


def _build_sentence_html(text: str, sent_start: int, sent_end: int,
                         highlight_spans: List[Tuple[int, int, str, str, str]]) -> str:
    """
    Build highlighted HTML for a single sentence.
    highlight_spans: list of (abs_start, abs_end, css_class, label, triple_key)
    """
    # Filter spans to this sentence and resolve overlaps (greedy left-to-right)
    local_spans = [(s, e, cls, lbl, tk) for s, e, cls, lbl, tk in highlight_spans
                   if s >= sent_start and e <= sent_end]
    local_spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    final_spans = []
    last_end = sent_start
    for s, e, cls, lbl, tk in local_spans:
        if s >= last_end:
            final_spans.append((s, e, cls, lbl, tk))
            last_end = e

    parts = []
    cursor = sent_start

    for s, e, cls, lbl, tk in final_spans:
        # Text before span
        if cursor < s:
            parts.append(html.escape(text[cursor:s]))

        span_text = html.escape(text[s:e])
        escaped_label = html.escape(lbl)
        parts.append(
            f'<mark class="{cls}">'
            f'{span_text} <sub class="canon-label">[{escaped_label}]</sub>'
            f'</mark>'
        )
        cursor = e

    # Remaining text
    if cursor < sent_end:
        parts.append(html.escape(text[cursor:sent_end]))

    return "".join(parts)


def _build_triple_graph_html(triples_in_sentence: List[Dict]) -> str:
    """
    Build the right-margin triple annotation for a sentence.
    Groups triples by verb to allow multi-edge rendering.
    """
    # Group by verb token index
    by_verb: Dict[int, List[Dict]] = {}
    for t in triples_in_sentence:
        vid = t["verb_onset"]
        if vid not in by_verb:
            by_verb[vid] = []
        by_verb[vid].append(t)

    parts = []
    for vid, group in sorted(by_verb.items()):
        # Check if we can draw a multi-edge graph (same agent, same verb)
        agents = {t["agent_name"] for t in group}
        patients = {t["patient_name"] for t in group}
        verb_text = group[0]["verb_text"]
        verb_lemma = group[0]["verb_lemma"]
        negated = group[0]["negated"]

        neg_prefix = "¬" if negated else ""

        if len(agents) == 1 and len(patients) > 1:
            # One agent → multiple patients
            agent = list(agents)[0]
            parts.append('<div class="triple-graph">')
            parts.append(f'<span class="triple-node agent-node" data-triplekey="{html.escape(group[0]["triple_key"])}" data-field="agent">{html.escape(agent)}</span>')
            for t in group:
                parts.append(
                    f'<span class="triple-edge" data-triplekey="{html.escape(t["triple_key"])}" data-field="verb">'
                    f'—{neg_prefix}{html.escape(verb_text)}&nbsp[{html.escape(verb_lemma)}]→</span> '
                    f'<span class="triple-node patient-node" data-triplekey="{html.escape(t["triple_key"])}" data-field="patient">'
                    f'{html.escape(t["patient_name"])}</span><br>'
                )
            parts.append('</div>')
        elif len(patients) == 1 and len(agents) > 1:
            # Multiple agents → one patient
            patient = list(patients)[0]
            parts.append('<div class="triple-graph">')
            for t in group:
                parts.append(
                    f'<span class="triple-node agent-node" data-triplekey="{html.escape(t["triple_key"])}" data-field="agent">'
                    f'{html.escape(t["agent_name"])}</span> '
                    f'<span class="triple-edge" data-triplekey="{html.escape(t["triple_key"])}" data-field="verb">'
                    f'—{neg_prefix}{html.escape(verb_text)}&nbsp[{html.escape(verb_lemma)}]→</span> '
                )
            parts.append(
                f'<span class="triple-node patient-node" data-triplekey="{html.escape(group[0]["triple_key"])}" data-field="patient">'
                f'{html.escape(patient)}</span>'
            )
            parts.append('</div>')
        else:
            # Render each triple separately
            for t in group:
                neg_p = "¬" if t["negated"] else ""
                parts.append(
                    f'<div class="triple-graph">'
                    f'<span class="triple-node agent-node" data-triplekey="{html.escape(t["triple_key"])}" data-field="agent">'
                    f'{html.escape(t["agent_name"])}</span> '
                    f'<span class="triple-edge" data-triplekey="{html.escape(t["triple_key"])}" data-field="verb">'
                    f'—{neg_p}{html.escape(t["verb_text"])}&nbsp[{html.escape(verb_lemma)}]→</span> '
                    f'<span class="triple-node patient-node" data-triplekey="{html.escape(t["triple_key"])}" data-field="patient">'
                    f'{html.escape(t["patient_name"])}</span>'
                    f'</div>'
                )

    return "\n".join(parts)


def build_full_html(
    text: str,
    enriched_triples: List[Dict],
    triples_df: pd.DataFrame,
    text_range: Optional[Tuple[int, int]] = None,
    split_label: str = "",
) -> str:
    """
    Build the complete HTML document.

    :param text_range: If provided, only render text in [start, end) byte range.
    :param split_label: Label for the split (e.g. "Split 1 of 3").
    """
    range_start = text_range[0] if text_range else 0
    range_end = text_range[1] if text_range else len(text)

    # Group triples by sentence
    by_sentence: Dict[int, List[Dict]] = {}
    for t in enriched_triples:
        sid = t["sentence_id"]
        if sid not in by_sentence:
            by_sentence[sid] = []
        by_sentence[sid].append(t)

    # Collect all sentence boundaries (for sentences with triples)
    all_sent_bounds = {}
    for t in enriched_triples:
        sid = t["sentence_id"]
        all_sent_bounds[sid] = (t["sent_start"], t["sent_end"])

    # Build all highlight spans per sentence
    all_highlights: Dict[int, List] = {}
    for sid, tlist in by_sentence.items():
        all_highlights[sid] = _resolve_highlight_spans(tlist)

    # --- We render text in range, splitting by paragraphs ---
    # Find all paragraph breaks (double newlines)
    paragraphs = text.split("\n\n")

    # For each paragraph, figure out which sentences (with triples) it contains
    # by tracking byte offsets as we walk through paragraphs
    body_parts = []
    current_byte = 0

    for para_idx, para_text in enumerate(paragraphs):
        para_start = text.find(para_text, current_byte)
        if para_start == -1:
            para_start = current_byte
        para_end = para_start + len(para_text)
        current_byte = para_end + 2  # skip \n\n

        # Skip paragraphs outside the text range
        if para_end <= range_start or para_start >= range_end:
            continue

        # Check for markdown headings
        heading_match = re.match(r"^(#+)\s+(.*)$", para_text.strip(), re.DOTALL)
        if heading_match:
            hashes, content = heading_match.groups()
            h_level = min(len(hashes), 6)
            if h_level == 1:
                body_parts.append(
                    f'</details><details open><summary><span class="chapter-heading">'
                    f'{html.escape(content)}</span></summary>'
                )
            else:
                body_parts.append(f'<h{h_level}>{html.escape(content)}</h{h_level}>')
            continue

        # Find which triple-sentences fall within this paragraph
        sids_in_para = []
        for sid, (ss, se) in all_sent_bounds.items():
            if ss >= para_start and se <= para_end:
                sids_in_para.append(sid)
        sids_in_para.sort(key=lambda sid: all_sent_bounds[sid][0])

        if not sids_in_para:
            # No triples in this paragraph — render plain text
            p_html = html.escape(para_text).replace("\n", "<br>\n")
            body_parts.append(f'<div class="text-row"><div class="text-col"><p data-byte-start="{para_start}">{p_html}</p></div>'
                              f'<div class="triple-col"></div></div>')
            continue

        # Render paragraph with highlights, sentence by sentence
        # We walk through the paragraph text, rendering highlighted sentences
        # and plain text between them
        para_parts = []
        triple_parts = []
        cursor = para_start

        for sid in sids_in_para:
            ss, se = all_sent_bounds[sid]
            highlights = all_highlights.get(sid, [])
            triples_here = by_sentence.get(sid, [])

            # Plain text before this sentence
            if cursor < ss:
                para_parts.append(html.escape(text[cursor:ss]))

            # Highlighted sentence
            sent_html = _build_sentence_html(text, ss, se, highlights)
            para_parts.append(f'<span class="triple-sentence" data-sid="{sid}">{sent_html}</span>')

            # Triple annotation for right margin
            triple_html = _build_triple_graph_html(triples_here)
            triple_parts.append(triple_html)

            cursor = se

        # Remaining text after last highlighted sentence
        if cursor < para_end:
            para_parts.append(html.escape(text[cursor:para_end]))

        text_html = "".join(para_parts).replace("\n", "<br>\n")
        right_html = "\n".join(triple_parts)

        body_parts.append(
            f'<div class="text-row">'
            f'<div class="text-col"><p data-byte-start="{para_start}">{text_html}</p></div>'
            f'<div class="triple-col">{right_html}</div>'
            f'</div>'
        )

    # Remove leading </details> if present
    body_html = "\n".join(body_parts)
    if body_html.startswith("</details>"):
        body_html = body_html[len("</details>"):]
    body_html += "</details>"

    # Serialize all triples as JSON for the JS editor
    triples_json = []
    for t in enriched_triples:
        triples_json.append({
            "triple_key": t["triple_key"],
            "row_idx": t["row_idx"],
            "agent_name": t["agent_name"],
            "agent_cid": t["agent_cid"],
            "verb_text": t["verb_text"],
            "verb_lemma": t["verb_lemma"],
            "negated": t["negated"],
            "patient_name": t["patient_name"],
            "patient_cid": t["patient_cid"],
            "verb_index": t["verb_onset"],
        })

    # Build full original triples for export (all rows, not just enriched)
    original_triples_json = []
    for row_idx, row in triples_df.iterrows():
        verb_token_id = int(row["index"])
        agent_cid = int(row["canonical_id_left"])
        patient_cid = int(row["canonical_id_right"])
        triple_key = f"{agent_cid}_{patient_cid}_{verb_token_id}_{row['word']}"
        original_triples_json.append({
            "triple_key": triple_key,
            "row_idx": int(row_idx),
            "canonical_id_left": agent_cid,
            "name_left": str(row["name_left"]),
            "role_left": str(row["role_left"]),
            "word": str(row["word"]),
            "lemma": str(row["lemma"]),
            "index": verb_token_id,
            "negated": bool(row["negated"]),
            "gender_left": str(row["gender_left"]),
            "canonical_id_right": patient_cid,
            "name_right": str(row["name_right"]),
            "role_right": str(row["role_right"]),
            "gender_right": str(row["gender_right"]),
        })

    triples_data_js = json.dumps(original_triples_json, ensure_ascii=False)

    title_suffix = f" — {split_label}" if split_label else ""
    subtitle_extra = f'<br><strong>{html.escape(split_label)}</strong>' if split_label else ""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Triple Verification{html.escape(title_suffix)}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.8;
            margin: 20px;
            padding: 0;
            color: #333;
            background-color: #fcfcfc;
        }}
        h1 {{
            text-align: center;
            color: #444;
            margin-bottom: 0.3em;
        }}
        .subtitle {{
            text-align: center;
            font-size: 0.9em;
            color: #666;
            margin-bottom: 20px;
        }}
        .controls {{
            text-align: center;
            margin: 0;
            padding: 12px 0;
            position: sticky;
            top: 0;
            z-index: 90;
            background: #fcfcfc;
            border-bottom: 1px solid #eee;
        }}
        .page-content {{
            padding-top: 10px;
        }}
        .controls button {{
            padding: 8px 20px;
            font-size: 0.95em;
            border: 1px solid #4a90d9;
            background: #4a90d9;
            color: white;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 5px;
        }}
        .controls button:hover {{
            background: #357abd;
        }}
        .controls .stats {{
            display: inline-block;
            margin-left: 15px;
            font-size: 0.85em;
            color: #666;
        }}
        .text-row {{
            display: flex;
            align-items: flex-start;
            gap: 20px;
            margin-bottom: 0.5em;
        }}
        .text-col {{
            flex: 0 0 65%;
            max-width: 65%;
            text-align: justify;
        }}
        .text-col p {{
            margin: 0 0 0.5em 0;
        }}
        .triple-col {{
            flex: 0 0 30%;
            max-width: 30%;
            font-size: 0.82em;
            color: #555;
            border-left: 2px solid #e0e0e0;
            padding-left: 12px;
            min-height: 1em;
        }}
        /* Highlight styles */
        mark.hl-agent, mark.hl-patient, mark.hl-verb {{
            background-color: #d4edda;
            color: #155724;
            border-radius: 3px;
            padding: 1px 3px;
            border: 1px solid rgba(21, 87, 36, 0.15);
            box-decoration-break: clone;
            -webkit-box-decoration-break: clone;
        }}
        mark.hl-verb {{
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid rgba(133, 100, 4, 0.15);
            text-decoration: underline;
            text-decoration-color: #e8a0b5;
            text-underline-offset: 3px;
        }}
        sub.canon-label {{
            color: #666;
            font-size: 0.7em;
            font-weight: 600;
        }}
        /* Triple graph styles */
        .triple-graph {{
            margin: 4px 0;
            padding: 4px 6px;
            background: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #e9ecef;
            line-height: 1.6;
        }}
        .triple-node {{
            display: inline-block;
            padding: 1px 6px;
            border-radius: 3px;
            cursor: pointer;
            border: 1px solid transparent;
        }}
        .triple-node:hover {{
            border: 1px dashed #4a90d9;
        }}
        .agent-node {{
            background: #d4edda;
            color: #155724;
            font-weight: 600;
        }}
        .patient-node {{
            background: #d4edda;
            color: #155724;
            font-weight: 600;
        }}
        .triple-edge {{
            color: #856404;
            font-style: italic;
            cursor: pointer;
            padding: 0 2px;
        }}
        .triple-edge:hover {{
            background: #fff3cd;
            border-radius: 3px;
        }}
        /* Editable state */
        .editing {{
            outline: 2px solid #4a90d9;
            background: white !important;
            min-width: 60px;
        }}
        .edited {{
            border-bottom: 2px solid #e83e8c !important;
            position: relative;
        }}
        .undo-btn {{
            display: inline-block;
            margin-left: 3px;
            cursor: pointer;
            color: #dc3545;
            font-size: 0.75em;
            font-weight: bold;
            vertical-align: super;
            opacity: 0.7;
            user-select: none;
        }}
        .undo-btn:hover {{
            opacity: 1;
            color: #a71d2a;
        }}
        /* Collapsible chapters */
        details {{
            margin-bottom: 0.5em;
            border-top: 1px solid #ddd;
        }}
        details[open] {{
            padding-bottom: 0.5em;
        }}
        summary {{
            list-style: none;
            cursor: pointer;
            padding: 8px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        summary::-webkit-details-marker {{ display: none; }}
        summary::before {{
            content: "▶";
            font-size: 0.65em;
            color: #888;
            transition: transform 0.2s;
            flex-shrink: 0;
        }}
        details[open] > summary::before {{
            transform: rotate(90deg);
        }}
        .chapter-heading {{
            font-size: 1.6em;
            font-weight: bold;
            color: #444;
        }}
        /* ---- Add-triple mode ---- */
        #add-mode-bar {{
            display: none;
            position: sticky;
            top: 0;
            z-index: 100;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            text-align: center;
            font-size: 0.95em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            border-radius: 0 0 8px 8px;
        }}
        #add-mode-bar.active {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }}
        #add-mode-bar .step-label {{
            font-weight: 700;
        }}
        #add-mode-bar button {{
            padding: 4px 14px;
            border: 1px solid rgba(255,255,255,0.5);
            background: rgba(255,255,255,0.15);
            color: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
        }}
        #add-mode-bar button:hover {{
            background: rgba(255,255,255,0.3);
        }}
        .add-mode .text-col {{
            cursor: crosshair;
        }}
        .add-mode .text-col ::selection {{
            background: #c3e6cb;
        }}
        .pending-span {{
            background: #b8daff;
            border-radius: 2px;
            padding: 1px 2px;
        }}
        /* Canonical name popup */
        #canon-popup {{
            display: none;
            position: fixed;
            z-index: 200;
            background: white;
            border: 1px solid #ccc;
            border-radius: 6px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            padding: 12px 16px;
            min-width: 260px;
        }}
        #canon-popup label {{
            font-size: 0.9em;
            color: #555;
            display: block;
            margin-bottom: 4px;
        }}
        #canon-popup input {{
            width: 100%;
            padding: 6px 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 0.95em;
            margin-bottom: 8px;
        }}
        #canon-popup .popup-btns {{
            display: flex;
            gap: 8px;
            justify-content: flex-end;
        }}
        #canon-popup .popup-btns button {{
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }}
        #canon-popup .btn-confirm {{
            background: #28a745;
            color: white;
            border: none;
        }}
        #canon-popup .btn-cancel {{
            background: #f8f9fa;
            color: #333;
            border: 1px solid #ccc;
        }}
        .new-triple {{
            border-left: 3px solid #667eea !important;
            background: #f0f0ff !important;
        }}
        .new-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            font-size: 0.65em;
            padding: 1px 5px;
            border-radius: 3px;
            margin-right: 4px;
            vertical-align: middle;
            font-weight: 700;
        }}
    </style>
</head>
<body>
    <h1>Triple Verification{html.escape(title_suffix)}</h1>
    <p class="subtitle">
        Double-click any <span style="background:#d4edda; padding:2px 6px; border-radius:3px;">agent</span> /
        <span style="background:#fff3cd; padding:2px 6px; border-radius:3px; text-decoration:underline; text-decoration-color:#e8a0b5;">verb</span> /
        <span style="background:#d4edda; padding:2px 6px; border-radius:3px;">patient</span>
        in the right column to edit. Download corrections when done.
        {subtitle_extra}
    </p>
    <div class="controls">
        <button onclick="downloadCorrections()">⬇ Download Corrections TSV</button>
        <button onclick="enterAddMode()" id="add-triple-btn" style="background:#28a745; border-color:#28a745;">＋ Add Triple</button>
        <span class="stats" id="edit-stats">0 edits made</span>
    </div>
    <div id="add-mode-bar">
        <span class="step-label" id="add-step-label">Step 1/3 — Select AGENT span</span>
        <span id="add-step-preview"></span>
        <button onclick="cancelAddMode()">✕ Cancel</button>
    </div>
    <div id="canon-popup">
        <label id="canon-popup-label">Canonical character name:</label>
        <input type="text" id="canon-popup-input" placeholder="e.g. Vera Claythorne">
        <div class="popup-btns">
            <button class="btn-cancel" onclick="cancelCanonPopup()">Cancel</button>
            <button class="btn-confirm" onclick="confirmCanonPopup()">Confirm</button>
        </div>
    </div>
    <div class="page-content">
    {body_html}
    </div>

<script>
// ============================
// Triple data
// ============================
const allTriples = {triples_data_js};

// Track edits: triple_key -> {{field: newValue, ...}}
const edits = {{}};
let editCount = 0;

// ============================
// Editing
// ============================
document.addEventListener('dblclick', function(e) {{
    const node = e.target.closest('.triple-node, .triple-edge');
    if (!node || node.contentEditable === 'true') return;

    const tripleKey = node.dataset.triplekey;
    const field = node.dataset.field;
    if (!tripleKey || !field) return;

    // Make editable
    node.contentEditable = 'true';
    node.classList.add('editing');
    node.focus();

    // Select text
    const range = document.createRange();
    range.selectNodeContents(node);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);

    // Store original value
    const originalText = node.textContent.replace(/^[—¬]+/, '').replace(/→$/, '').trim();
    node.dataset.original = originalText;

    function finishEdit() {{
        node.contentEditable = 'false';
        node.classList.remove('editing');

        let newText = node.textContent.trim();

        // For edge, strip formatting chars
        if (field === 'verb') {{
            newText = newText.replace(/^[—¬]+/, '').replace(/→$/, '').trim();
        }}

        const original = node.dataset.original;
        if (newText !== original && newText !== '') {{
            // Record edit
            if (!edits[tripleKey]) edits[tripleKey] = {{}};
            edits[tripleKey][field] = {{
                original: original,
                corrected: newText
            }};
            node.classList.add('edited');
            editCount++;
            document.getElementById('edit-stats').textContent = editCount + ' edit(s) made';

            // Restore formatting for verb edge
            if (field === 'verb') {{
                const triple = allTriples.find(t => t.triple_key === tripleKey);
                const negPrefix = (triple && triple.negated) ? '¬' : '';
                node.textContent = '—' + negPrefix + newText + '→';
            }}

            // Add undo button
            addUndoBtn(node, tripleKey, field, original);
        }} else {{
            // Restore original formatting for verb edge
            if (field === 'verb') {{
                const triple = allTriples.find(t => t.triple_key === tripleKey);
                const negPrefix = (triple && triple.negated) ? '¬' : '';
                node.textContent = '—' + negPrefix + original + '→';
            }}
        }}
    }}

    node.addEventListener('blur', finishEdit, {{ once: true }});
    node.addEventListener('keydown', function(ev) {{
        if (ev.key === 'Enter') {{
            ev.preventDefault();
            node.blur();
        }}
    }});
}});

// ============================
// Undo edits
// ============================
function addUndoBtn(node, tripleKey, field, originalValue) {{
    // Remove existing undo btn for this node if any
    const existing = node.parentElement.querySelector(`.undo-btn[data-triplekey="${{tripleKey}}"][data-field="${{field}}"]`);
    if (existing) existing.remove();

    const btn = document.createElement('span');
    btn.className = 'undo-btn';
    btn.textContent = '✕';
    btn.title = 'Revert this edit';
    btn.dataset.triplekey = tripleKey;
    btn.dataset.field = field;
    btn.addEventListener('click', function(e) {{
        e.stopPropagation();
        // Restore original value
        if (field === 'verb') {{
            const triple = allTriples.find(t => t.triple_key === tripleKey);
            const negPrefix = (triple && triple.negated) ? '¬' : '';
            node.textContent = '—' + negPrefix + originalValue + '→';
        }} else {{
            node.textContent = originalValue;
        }}
        node.classList.remove('edited');

        // Remove edit from tracking
        if (edits[tripleKey] && edits[tripleKey][field]) {{
            delete edits[tripleKey][field];
            if (Object.keys(edits[tripleKey]).length === 0) {{
                delete edits[tripleKey];
            }}
            editCount = Math.max(0, editCount - 1);
            document.getElementById('edit-stats').textContent = editCount + ' edit(s) made';
        }}

        // Remove the undo button itself
        btn.remove();
    }});
    node.after(btn);
}}

// ============================
// Download corrections (with new triples)
// ============================
function downloadCorrections() {{
    const header = [
        'triple_key', 'row_idx',
        'canonical_id_left', 'name_left', 'role_left',
        'word', 'lemma', 'index', 'negated',
        'gender_left',
        'canonical_id_right', 'name_right', 'role_right',
        'gender_right',
        'corrected_agent', 'corrected_verb', 'corrected_patient',
        'changed', 'changes_applied',
        'is_new',
        'agent_byte_start', 'agent_byte_end',
        'verb_byte_start', 'verb_byte_end',
        'patient_byte_start', 'patient_byte_end'
    ].join('\\t');

    const rows = [header];

    // Existing triples
    for (const triple of allTriples) {{
        const tk = triple.triple_key;
        const edit = edits[tk] || {{}};
        const changed = Object.keys(edit).length > 0;

        const correctedAgent = edit.agent ? edit.agent.corrected : triple.name_left;
        const correctedVerb = edit.verb ? edit.verb.corrected : triple.word;
        const correctedPatient = edit.patient ? edit.patient.corrected : triple.name_right;

        const changes = [];
        if (edit.agent) changes.push('agent: "' + edit.agent.original + '" \u2192 "' + edit.agent.corrected + '"');
        if (edit.verb) changes.push('verb: "' + edit.verb.original + '" \u2192 "' + edit.verb.corrected + '"');
        if (edit.patient) changes.push('patient: "' + edit.patient.original + '" \u2192 "' + edit.patient.corrected + '"');
        const changesStr = changes.join('; ');

        const row = [
            tk, triple.row_idx,
            triple.canonical_id_left, triple.name_left, 'agent',
            triple.word, triple.lemma, triple.index, triple.negated,
            triple.gender_left,
            triple.canonical_id_right, triple.name_right, 'patient',
            triple.gender_right,
            correctedAgent, correctedVerb, correctedPatient,
            changed, changesStr,
            'false', '', '', '', '', '', ''
        ].join('\\t');

        rows.push(row);
    }}

    // New triples
    for (const nt of newTriples) {{
        const row = [
            nt.triple_key, nt.row_idx,
            '', nt.agent_canon, 'agent',
            nt.verb_text, nt.verb_text, '', 'false',
            '',
            '', nt.patient_canon, 'patient',
            '',
            nt.agent_canon, nt.verb_text, nt.patient_canon,
            'true', 'new triple',
            'true',
            nt.agent_byte_start, nt.agent_byte_end,
            nt.verb_byte_start, nt.verb_byte_end,
            nt.patient_byte_start, nt.patient_byte_end
        ].join('\\t');

        rows.push(row);
    }}

    const tsv = rows.join('\\n');
    const blob = new Blob([tsv], {{ type: 'text/tab-separated-values' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'triples_corrections.tsv';
    a.click();
    URL.revokeObjectURL(url);
}}

// ============================
// Add triple mode
// ============================
let addMode = false;
let addStep = 0; // 0 = off, 1 = agent, 2 = verb, 3 = patient
const addData = {{ agent: null, verb: null, patient: null }};
const newTriples = [];

function enterAddMode() {{
    addMode = true;
    addStep = 1;
    addData.agent = null;
    addData.verb = null;
    addData.patient = null;
    document.body.classList.add('add-mode');
    const bar = document.getElementById('add-mode-bar');
    bar.classList.add('active');
    updateAddStepLabel();
}}

function cancelAddMode() {{
    addMode = false;
    addStep = 0;
    document.body.classList.remove('add-mode');
    document.getElementById('add-mode-bar').classList.remove('active');
    document.getElementById('add-step-preview').textContent = '';
    hideCanonPopup();
}}

function updateAddStepLabel() {{
    const label = document.getElementById('add-step-label');
    const preview = document.getElementById('add-step-preview');
    const steps = ['', 'Select AGENT span', 'Select VERB span', 'Select PATIENT span'];
    label.textContent = 'Step ' + addStep + '/3 — ' + steps[addStep];

    // Show accumulated selections
    const parts = [];
    if (addData.agent) parts.push('Agent: "' + addData.agent.text + '" [' + addData.agent.canon + ']');
    if (addData.verb) parts.push('Verb: "' + addData.verb.text + '"');
    preview.textContent = parts.length ? '  |  ' + parts.join('  •  ') : '';
}}

// Capture text selection in add mode
document.addEventListener('mouseup', function(e) {{
    if (!addMode || addStep < 1 || addStep > 3) return;

    // Only capture from .text-col elements
    const textCol = e.target.closest('.text-col');
    if (!textCol) return;

    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return;

    const selectedText = sel.toString().trim();
    if (!selectedText) return;

    // Compute byte offset: find enclosing <p> with data-byte-start
    const range = sel.getRangeAt(0);
    const pElem = range.startContainer.nodeType === 3
        ? range.startContainer.parentElement.closest('p[data-byte-start]')
        : range.startContainer.closest('p[data-byte-start]');
    if (!pElem) return;

    const paraByteStart = parseInt(pElem.dataset.byteStart, 10);

    // Walk text nodes inside the <p> to compute the character offset
    const walker = document.createTreeWalker(pElem, NodeFilter.SHOW_TEXT, null, false);
    let charOffset = 0;
    let found = false;
    while (walker.nextNode()) {{
        const node = walker.currentNode;
        if (node === range.startContainer) {{
            charOffset += range.startOffset;
            found = true;
            break;
        }}
        charOffset += node.textContent.length;
    }}
    if (!found) return;

    // Compute absolute byte positions.
    // The original text was split by double-newlines to form paragraphs, then escaped.
    // pElem.textContent gives us the plain text of the paragraph (HTML tags stripped).
    // We use the TextEncoder to get byte-accurate offsets since the raw text may have
    // multi-byte characters.
    const paraPlainText = pElem.textContent;
    const encoder = new TextEncoder();
    const bytesBeforeSelection = encoder.encode(paraPlainText.substring(0, charOffset)).length;
    const bytesOfSelection = encoder.encode(selectedText).length;

    const absByteStart = paraByteStart + bytesBeforeSelection;
    const absByteEnd = absByteStart + bytesOfSelection;

    const spanInfo = {{
        text: selectedText,
        byte_start: absByteStart,
        byte_end: absByteEnd,
    }};

    // Agent or patient → need canonical name
    if (addStep === 1 || addStep === 3) {{
        // Store pending span and show popup
        window._pendingSpanInfo = spanInfo;
        window._pendingField = addStep === 1 ? 'agent' : 'patient';
        showCanonPopup(e.clientX, e.clientY, addStep === 1 ? 'agent' : 'patient');
    }} else if (addStep === 2) {{
        // Verb — no canonical name needed
        addData.verb = spanInfo;
        addStep = 3;
        updateAddStepLabel();
        sel.removeAllRanges();
    }}
}});

function showCanonPopup(x, y, role) {{
    const popup = document.getElementById('canon-popup');
    const label = document.getElementById('canon-popup-label');
    const input = document.getElementById('canon-popup-input');
    label.textContent = 'Canonical ' + role + ' name:';
    input.value = '';
    popup.style.display = 'block';
    popup.style.left = Math.min(x, window.innerWidth - 300) + 'px';
    popup.style.top = Math.min(y + 10, window.innerHeight - 120) + 'px';
    input.focus();
}}

function hideCanonPopup() {{
    document.getElementById('canon-popup').style.display = 'none';
    window._pendingSpanInfo = null;
    window._pendingField = null;
}}

function cancelCanonPopup() {{
    hideCanonPopup();
}}

function confirmCanonPopup() {{
    const input = document.getElementById('canon-popup-input');
    const canonName = input.value.trim();
    if (!canonName) {{
        input.style.borderColor = 'red';
        return;
    }}
    input.style.borderColor = '#ccc';

    const spanInfo = window._pendingSpanInfo;
    const field = window._pendingField;
    if (!spanInfo || !field) return;

    spanInfo.canon = canonName;
    addData[field] = spanInfo;
    hideCanonPopup();
    window.getSelection().removeAllRanges();

    if (field === 'agent') {{
        addStep = 2;
    }} else if (field === 'patient') {{
        // All three collected → finalize
        finalizeNewTriple();
        return;
    }}
    updateAddStepLabel();
}}

// Enter key confirms popup
document.getElementById('canon-popup-input').addEventListener('keydown', function(e) {{
    if (e.key === 'Enter') {{
        e.preventDefault();
        confirmCanonPopup();
    }} else if (e.key === 'Escape') {{
        cancelCanonPopup();
    }}
}});

function finalizeNewTriple() {{
    const tripleKey = 'new_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
    const newTriple = {{
        triple_key: tripleKey,
        row_idx: -1,
        is_new: true,
        // Agent
        agent_text: addData.agent.text,
        agent_byte_start: addData.agent.byte_start,
        agent_byte_end: addData.agent.byte_end,
        agent_canon: addData.agent.canon,
        // Verb
        verb_text: addData.verb.text,
        verb_byte_start: addData.verb.byte_start,
        verb_byte_end: addData.verb.byte_end,
        // Patient
        patient_text: addData.patient.text,
        patient_byte_start: addData.patient.byte_start,
        patient_byte_end: addData.patient.byte_end,
        patient_canon: addData.patient.canon,
    }};
    newTriples.push(newTriple);

    // Insert visual triple into the closest .triple-col
    const verbByte = addData.verb.byte_start;
    let bestRow = null;
    let bestDist = Infinity;
    document.querySelectorAll('.text-col p[data-byte-start]').forEach(p => {{
        const bs = parseInt(p.dataset.byteStart, 10);
        const dist = Math.abs(bs - verbByte);
        if (dist < bestDist) {{
            bestDist = dist;
            bestRow = p.closest('.text-row');
        }}
    }});

    if (bestRow) {{
        const tripleCol = bestRow.querySelector('.triple-col');
        if (tripleCol) {{
            const div = document.createElement('div');
            div.className = 'triple-graph new-triple';
            div.innerHTML =
                '<span class="new-badge">NEW</span> ' +
                '<span class="triple-node agent-node">' + escHtml(addData.agent.canon) + '</span> ' +
                '<span class="triple-edge">—' + escHtml(addData.verb.text) + '→</span> ' +
                '<span class="triple-node patient-node">' + escHtml(addData.patient.canon) + '</span>';
            tripleCol.appendChild(div);
        }}
    }}

    editCount++;
    document.getElementById('edit-stats').textContent = editCount + ' edit(s) made';
    cancelAddMode();
}}

function escHtml(str) {{
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}}


</script>
</body>
</html>"""

    return html_content


# ============================================================================
# MAIN
# ============================================================================

def main(
    text_path: Path = RAW_TEXT,
    tokens_path: Path = TOKENS,
    triples_path: Path = None,
    coref_dir: Path = COREF_OUT,
    out_dir: Path = OUT_DIR,
    out_html: Path = None,
    n_splits: int = 1,
):
    """Run the triple verification HTML generator."""
    print_headers("TRIPLE VERIFICATION — HTML GENERATOR", "=", prefix="\n")

    # Resolve default paths
    if triples_path is None:
        triples_path = out_dir / "obj_extraction" / "triples.csv"
    if out_html is None:
        out_html = out_dir / "obj_extraction" / "verify_triples.html"
    # If out_html is a directory, append default filename
    if out_html.is_dir():
        out_html = out_html / "verify_triples.html"

    # Step 1: Load data
    print_information("Loading data...", 1, "\n")

    text = load_text(text_path)
    print_information(f"Loaded text: {len(text)} chars", prefix="    ")

    triples_df = load_triples(triples_path)
    print_information(f"Loaded {len(triples_df)} triples", prefix="    ")

    tokens_df = load_tokens(tokens_path)
    print_information(f"Loaded {len(tokens_df)} tokens", prefix="    ")

    span_index = load_span_index(coref_dir / "span_index.jsonl")
    print_information(f"Loaded {len(span_index)} coreference spans", prefix="    ")

    # Step 2: Build lookup structures
    print_information("Building lookup structures...", 2, "\n")

    token_index = build_token_index(tokens_df)
    sentence_bounds = build_sentence_boundaries(tokens_df)
    span_lookup = build_span_lookup_by_canonical(span_index)

    print_information(f"Token index: {len(token_index)} entries", prefix="    ")
    print_information(f"Sentence bounds: {len(sentence_bounds)} sentences", prefix="    ")
    print_information(f"Span lookup: {len(span_lookup)} canonical IDs", prefix="    ")

    # Step 3: Enrich triples with byte offsets
    print_information("Enriching triples with byte offsets...", 3, "\n")

    enriched = enrich_triples(triples_df, token_index, sentence_bounds, span_lookup)
    n_with_agent = sum(1 for t in enriched if t["agent_span"] is not None)
    n_with_patient = sum(1 for t in enriched if t["patient_span"] is not None)
    print_information(f"Enriched {len(enriched)} triples", prefix="    ")
    print_information(f"  Agent spans found: {n_with_agent}/{len(enriched)}", prefix="    ")
    print_information(f"  Patient spans found: {n_with_patient}/{len(enriched)}", prefix="    ")

    # Step 4: Build HTML
    print_information("Generating HTML...", 4, "\n")

    if n_splits <= 1:
        # Single file mode
        html_content = build_full_html(text, enriched, triples_df)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(html_content, encoding="utf-8")
        print_information(f"Written to → {out_html}", "✓", col="GREEN")
        print_information(f"Open in browser: file://{out_html.absolute()}", prefix="    ")
    else:
        # Split mode
        chapter_bounds = find_chapter_boundaries(text)
        print_information(f"Found {len(chapter_bounds)} chapter boundaries", prefix="    ")

        split_ranges = compute_split_ranges(enriched, chapter_bounds, len(text), n_splits)
        print_information(f"Computed {len(split_ranges)} splits", prefix="    ")

        out_html.parent.mkdir(parents=True, exist_ok=True)
        stem = out_html.stem
        suffix = out_html.suffix or ".html"

        for i, (rng_start, rng_end, split_triples) in enumerate(split_ranges, 1):
            split_label = f"Split {i} of {len(split_ranges)}"

            # Build a DataFrame subset for the triples in this split
            split_row_idxs = {t["row_idx"] for t in split_triples}
            split_df = triples_df.loc[triples_df.index.isin(split_row_idxs)]

            html_content = build_full_html(
                text, split_triples, split_df,
                text_range=(rng_start, rng_end),
                split_label=split_label,
            )

            split_path = out_html.parent / f"{stem}_{i}{suffix}"
            split_path.write_text(html_content, encoding="utf-8")
            print_information(
                f"Split {i}: {len(split_triples)} triples, "
                f"text range [{rng_start}:{rng_end}] → {split_path.name}",
                "✓", col="GREEN"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Triple Verification HTML")
    parser.add_argument("--text", type=Path, default=RAW_TEXT,
                        help="Path to raw text file")
    parser.add_argument("--tokens", type=Path, default=TOKENS,
                        help="Path to .tokens file")
    parser.add_argument("--triples", type=Path, default=None,
                        help="Path to triples.csv")
    parser.add_argument("--coref-dir", type=Path, default=COREF_OUT,
                        help="Directory containing span_index.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Output directory")
    parser.add_argument("--out-html", type=Path, default=None,
                        help="Output HTML file path")
    parser.add_argument("--split", type=int, default=1,
                        help="Number of splits to divide triples into (default: 1 = no split)")
    args = parser.parse_args()

    main(
        text_path=args.text,
        tokens_path=args.tokens,
        triples_path=args.triples,
        coref_dir=args.coref_dir,
        out_dir=args.out_dir,
        out_html=args.out_html,
        n_splits=args.split,
    )
