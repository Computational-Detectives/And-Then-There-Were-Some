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
import csv
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import OUT_DIR, RAW_TEXT, TOKENS, COREF_OUT, PRONOUNS
from .utils import load_text, load_span_index, print_information, print_headers


# ============================================================================
# DATA LOADING
# ============================================================================

def load_triples(triples_path: Path) -> pd.DataFrame:
    """Load triples.csv (TSV)."""
    df = pd.read_csv(triples_path, sep="\t", keep_default_na=False)
    return df


def load_tokens(tokens_path: Path) -> pd.DataFrame:
    """Load the .tokens file as a DataFrame."""
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
        agents = set(t["agent_name"] for t in group)
        patients = set(t["patient_name"] for t in group)
        verb_text = group[0]["verb_text"]
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
                    f'—{neg_prefix}{html.escape(verb_text)}→</span> '
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
                    f'—{neg_prefix}{html.escape(verb_text)}→</span> '
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
                    f'—{neg_p}{html.escape(t["verb_text"])}→</span> '
                    f'<span class="triple-node patient-node" data-triplekey="{html.escape(t["triple_key"])}" data-field="patient">'
                    f'{html.escape(t["patient_name"])}</span>'
                    f'</div>'
                )

    return "\n".join(parts)


def build_full_html(text: str, enriched_triples: List[Dict], triples_df: pd.DataFrame) -> str:
    """Build the complete HTML document."""

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

    # --- We render the full text, splitting by paragraphs ---
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
            body_parts.append(f'<div class="text-row"><div class="text-col"><p>{p_html}</p></div>'
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
            f'<div class="text-col"><p>{text_html}</p></div>'
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

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Triple Verification</title>
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
            margin: 15px 0 25px 0;
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
    </style>
</head>
<body>
    <h1>Triple Verification</h1>
    <p class="subtitle">
        Double-click any <span style="background:#d4edda; padding:2px 6px; border-radius:3px;">agent</span> /
        <span style="background:#fff3cd; padding:2px 6px; border-radius:3px; text-decoration:underline; text-decoration-color:#e8a0b5;">verb</span> /
        <span style="background:#d4edda; padding:2px 6px; border-radius:3px;">patient</span>
        in the right column to edit. Download corrections when done.
    </p>
    <div class="controls">
        <button onclick="downloadCorrections()">⬇ Download Corrections TSV</button>
        <span class="stats" id="edit-stats">0 edits made</span>
    </div>
    <hr style="margin: 20px 0; border: 0; border-top: 1px solid #eee;">
    {body_html}

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
// Download corrections
// ============================
function downloadCorrections() {{
    // Build TSV with ALL triples, marking which were changed
    const header = [
        'triple_key', 'row_idx',
        'canonical_id_left', 'name_left', 'role_left',
        'word', 'lemma', 'index', 'negated',
        'gender_left',
        'canonical_id_right', 'name_right', 'role_right',
        'gender_right',
        'corrected_agent', 'corrected_verb', 'corrected_patient',
        'changed', 'changes_applied'
    ].join('\\t');

    const rows = [header];

    for (const triple of allTriples) {{
        const tk = triple.triple_key;
        const edit = edits[tk] || {{}};
        const changed = Object.keys(edit).length > 0;

        const correctedAgent = edit.agent ? edit.agent.corrected : triple.name_left;
        const correctedVerb = edit.verb ? edit.verb.corrected : triple.word;
        const correctedPatient = edit.patient ? edit.patient.corrected : triple.name_right;

        // Build changes log
        const changes = [];
        if (edit.agent) changes.push('agent: "' + edit.agent.original + '" → "' + edit.agent.corrected + '"');
        if (edit.verb) changes.push('verb: "' + edit.verb.original + '" → "' + edit.verb.corrected + '"');
        if (edit.patient) changes.push('patient: "' + edit.patient.original + '" → "' + edit.patient.corrected + '"');
        const changesStr = changes.join('; ');

        const row = [
            tk, triple.row_idx,
            triple.canonical_id_left, triple.name_left, 'agent',
            triple.word, triple.lemma, triple.index, triple.negated,
            triple.gender_left,
            triple.canonical_id_right, triple.name_right, 'patient',
            triple.gender_right,
            correctedAgent, correctedVerb, correctedPatient,
            changed, changesStr
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

    html_content = build_full_html(text, enriched, triples_df)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_content, encoding="utf-8")
    print_information(f"Written to → {out_html}", "✓", col="GREEN")
    print_information(f"Open in browser: file://{out_html.absolute()}", prefix="    ")


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
    args = parser.parse_args()

    main(
        text_path=args.text,
        tokens_path=args.tokens,
        triples_path=args.triples,
        coref_dir=args.coref_dir,
        out_dir=args.out_dir,
        out_html=args.out_html,
    )
