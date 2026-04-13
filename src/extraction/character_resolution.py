from __future__ import annotations

import json

from tqdm import tqdm
from pathlib import Path
from typing import Any, Tuple, List, Dict

from ..config import (
    RAW_TEXT, CLEAN_NAMES, OUT_DIR, CHAR_RES_OUT,
    FUZZY_AUTO_ACCEPT, FUZZY_REVIEW_LOW, TOKENS
)

from .utils import (
    build_variant_index, match_name,
    print_headers, print_information,
    load_spacy_model, load_gliner_model,
    load_text, snap_span, log_print
)


# ============================================================================
# SAVE
# ============================================================================

def save_ner(
    alias_dict: Dict,
    review_items: List[Dict],
    ner_spans: List[Dict],
    stage_dir: Path,
) -> None:
    """Write Stage 1 outputs to disk."""
    stage_dir.mkdir(parents=True, exist_ok=True)

    # alias_dict.json
    with open(stage_dir / "alias_dict.json", "w", encoding="utf-8") as f:
        json.dump(alias_dict, f, indent=2, ensure_ascii=False)

    # review_items.csv
    import pandas as pd
    pd.DataFrame(review_items).to_csv(stage_dir / "review_items.csv", index=False)

    # ner_spans.jsonl
    with open(stage_dir / "ner_spans.jsonl", "w", encoding="utf-8") as f:
        for span in ner_spans:
            f.write(json.dumps(span, ensure_ascii=False) + "\n")


# ============================================================================
# VERBOSE STATISTICS
# ============================================================================

def print_verbose_stats(all_spans, alias_dict, review_items, clean_names):
    from collections import Counter

    print_headers("STAGE 1 — VERBOSE STATISTICS", "-", prefix="\n")

    # NER source breakdown
    source_counts = Counter(s["source"] for s in all_spans)
    log_print("    NER Source Breakdown:")
    for src, count in source_counts.most_common():
        pct = count / len(all_spans) * 100 if all_spans else 0
        log_print(f"      {src:15s} {count:5d} spans  ({pct:5.1f}%)")

    # Per-character coverage: how many spans matched each canonical ID
    char_counts = Counter()
    for form, info in alias_dict.items():
        char_counts[info["fullname"]] += sum(
            1 for s in all_spans if s["text"] == form
        )
    log_print("\n    Per-Character Span Counts (auto-accepted):")
    for name, count in char_counts.most_common():
        log_print(f"      {name:35s} {count:5d} spans")

    # Canonical coverage check
    all_canonical = set(clean_names["fullname"].tolist())
    matched_canonical = {info["fullname"] for info in alias_dict.values()}
    missing = all_canonical - matched_canonical
    if missing:
        log_print(f"\n    ⚠ Missing canonical characters ({len(missing)}):")
        for name in sorted(missing):
            log_print(f"      - {name}")
    else:
        log_print(f"\n    ✓ All {len(all_canonical)} canonical characters found in alias dict")

    # Match score distribution
    scores = [info["score"] for info in alias_dict.values()]
    if scores:
        log_print(f"\n    Match Score Distribution:")
        log_print(f"      Min: {min(scores):.1f}  Max: {max(scores):.1f}  "
                f"Mean: {sum(scores)/len(scores):.1f}  "
                f"Median: {sorted(scores)[len(scores)//2]:.1f}")

    # Review items summary
    review_statuses = Counter(r["status"] for r in review_items)
    if review_items:
        log_print(f"\n    Review Items by Status:")
        for status, count in review_statuses.most_common():
            log_print(f"      {status:15s} {count:5d}")

    # Full alias dictionary
    print_headers("ALIAS DICTIONARY", "-", prefix="\n")
    for form, info in sorted(alias_dict.items(), key=lambda x: x[1]["canonical_id"]):
        log_print(f"    {form:30s} → {info['fullname']:30s} (score: {info['score']:.1f})")


# ============================================================================
# SENTENCE SEGMENTATION
# ============================================================================

def segment_sentences(text: str, tokens_path: Path) -> List[Dict]:
    """
    Segment text into sentences by reusing the .tokens file from the tokenisation step.

    :return: List of ``{ sid, text, start_char, end_char }``.
    """
    import pandas as pd
    df = pd.read_csv(tokens_path, sep="\t", keep_default_na=False)
    
    sentences = []
    for sid, group in df.groupby("sentence_ID"):
        start_char = int(group["byte_onset"].min())
        end_char = int(group["byte_offset"].max())
        sentences.append({
            "sid": int(sid),
            "text": text[start_char:end_char],
            "start_char": start_char,
            "end_char": end_char,
            "tokens": group.to_dict('records')
        })
    return sentences


# ============================================================================
# DICTIONARY PRE-PASS
# ============================================================================

def fuzzy_match_ngrams(
    sentences: List[Dict],
    clean_names: pd.DataFrame,
    variant_idx: Tuple[Dict, List[str], Dict, List[str]],
    text_ref: str = "",
) -> List[Dict]:
    """
    Scan sentences for n-gram fuzzy matches against name variants
    built from a list of clean names to catch character names that NER may have missed.

    :return: Additional PERSON spans.
    """
    variant_to_ids, all_variants, id_to_gender, all_name_tokens = variant_idx

    extra_spans: List[Dict] = []

    for sent in sentences:
        toks = sent.get("tokens", [])
        if not toks:
            continue
            
        words = [str(t["word"]) for t in toks]
        sid = sent["sid"]

        # Try n-grams of length 1 to 4
        for n in range(1, min(5, len(words) + 1)):
            for i in range(len(words) - n + 1):
                ngram_tokens_list = toks[i:i + n]
                ngram = " ".join(words[i:i + n])

                # Quick check: does any token overlap with known name tokens?
                ngram_lower = ngram.lower()
                ngram_tokens = set(ngram_lower.split())
                if not ngram_tokens & all_name_tokens:
                    continue
                
                # Check if the ngram is a known name
                cid, fullname, score, _ = match_name(
                    ngram, "u",  # gender unknown at this stage
                    variant_to_ids, all_variants, id_to_gender,
                    clean_names, all_name_tokens,
                    threshold=FUZZY_AUTO_ACCEPT,
                )

                if cid is not None and score >= FUZZY_AUTO_ACCEPT:
                    start_char = int(ngram_tokens_list[0]["byte_onset"])
                    end_char = int(ngram_tokens_list[-1]["byte_offset"])
                    # Use exact char slice as text representation
                    extra_spans.append({
                        "text": text_ref[start_char:end_char] if 'text_ref' in locals() else ngram,
                        "sid": sid,
                        "start_char": start_char,
                        "end_char": end_char,
                        "source": "dict_prepass",
                    })

    return extra_spans


# ============================================================================
# ALIAS DICTIONARY CONSTRUCTION
# ============================================================================

def build_alias_dict(
    ner_spans: List[Dict],
    clean_names: pd.DataFrame,
    variant_idx: Tuple[Dict, List[str], Dict, List[str]],
) -> Tuple[Dict, List[Dict]]:
    """
    Fuzzy-match each unique surface form to the canonical names CSV.

    :return: ``(alias_dict, review_items)``
        - alias_dict: ``{ surface_form: { canonical_id, fullname, score } }``
        - review_items: list of dicts for manual review
    """
    variant_to_ids, all_variants, id_to_gender, all_name_tokens = variant_idx

    # Collect unique surface forms
    unique_forms: Dict[str, List[int]] = {}
    for span in ner_spans:
        form = span["text"]
        if form not in unique_forms:
            unique_forms[form] = []
        unique_forms[form].append(span["sid"])

    alias_dict: Dict[str, Dict] = {}
    review_items: List[Dict] = []

    for form, sids in unique_forms.items():
        cid, fullname, score, _ = match_name(
            form, "u",
            variant_to_ids, all_variants, id_to_gender,
            clean_names, all_name_tokens,
            threshold=FUZZY_REVIEW_LOW,
        )

        if cid is not None and score >= FUZZY_AUTO_ACCEPT:
            alias_dict[form] = {
                "canonical_id": int(cid),
                "fullname": fullname,
                "score": round(score, 2),
            }
        elif cid is not None and score >= FUZZY_REVIEW_LOW:
            review_items.append({
                "surface_form": form,
                "candidate_id": int(cid),
                "candidate_name": fullname,
                "score": round(score, 2),
                "occurrences": len(sids),
                "status": "review",
            })
        else:
            review_items.append({
                "surface_form": form,
                "candidate_id": None,
                "candidate_name": None,
                "score": 0.0,
                "occurrences": len(sids),
                "status": "unresolved",
            })

    return alias_dict, review_items


# ============================================================================
# NAMED ENTITY RECOGNITION
# ============================================================================

# =========== NER HELPERS ===========
def _deduplicate_spans(spans: List[Dict]) -> List[Dict]:
    """
    Remove duplicate / overlapping PERSON spans.
    When two spans overlap, prefer GLiNER over spaCy.
    """
    # Sort by start_char, then by length descending
    spans.sort(key=lambda s: (s["start_char"], -(s["end_char"] - s["start_char"])))

    kept: List[Dict] = []
    for span in spans:
        # Check overlap with last kept span
        if kept and span["start_char"] < kept[-1]["end_char"]:
            # Overlap — keep the GLiNER one, or the longer one
            prev = kept[-1]
            if span["source"] == "gliner" and prev["source"] != "gliner":
                kept[-1] = span  # prefer GLiNER
            # Otherwise keep the existing (first / longer)
            continue
        kept.append(span)

    return kept


# =========== NER MAIN ===========
def run_ner(
    sentences: List[Dict],
    gliner_model: Any,
    nlp: Any,
    full_text: str
) -> List[Dict]:
    """
    Run GLiNER (primary) and spaCy NER (backup) to detect PERSON spans.

    :param sentences: Output of ``segment_sentences``.
    :param gliner_model: A loaded GLiNER model instance.
    :param nlp: A loaded spaCy model with NER.
    :return: List of ``{ text, sid, start_char, end_char, source }``.
    """
    spans: List[Dict] = []

    for sent in tqdm(sentences):
        sid = sent["sid"]
        sent_text = sent["text"]
        sent_start = sent["start_char"]
        toks = sent.get("tokens", [])

        # --- GLiNER ---
        try:
            gl_entities = gliner_model.predict_entities(
                sent_text,
                ["person", "character"],
                threshold=0.4,
            )

            for ent in gl_entities:
                raw_start = sent_start + ent["start"]
                raw_end = sent_start + ent["end"]
                s, e, fixed_text = snap_span(raw_start, raw_end, toks, full_text)
                spans.append({
                    "text": fixed_text,
                    "sid": sid,
                    "start_char": s,
                    "end_char": e,
                    "source": "gliner",
                })
        except Exception:
            pass  # GLiNER may fail on very short / empty sentences

        # --- spaCy NER backup ---
        if sent_text.strip():
            doc = nlp(sent_text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    raw_start = sent_start + ent.start_char
                    raw_end = sent_start + ent.end_char
                    s, e, fixed_text = snap_span(raw_start, raw_end, toks, full_text)
                    spans.append({
                        "text": fixed_text,
                        "sid": sid,
                        "start_char": s,
                        "end_char": e,
                        "source": "spacy",
                    })

    # Deduplicate overlapping spans — prefer GLiNER when both fire
    spans = _deduplicate_spans(spans)
    return spans


# ============================================================================
# MAIN
# ============================================================================

def main(
    text_path: Path = RAW_TEXT,
    names_csv: Path = CLEAN_NAMES,
    out_dir: Path = OUT_DIR,
    tokens_path: Path = TOKENS,
    verbose: bool = False,
    nlp=None,
    gliner_model=None,
) -> Tuple[Dict, List[Dict]]:
    """
    Stage 1 — Character Resolution

    Detect PERSON spans via GLiNER + spaCy NER, fuzzy-match against
    the canonical names CSV, and produce an alias dictionary mapping
    surface forms → canonical character IDs.
    """
    print_headers("STAGE 1 — CHARACTER RESOLUTION", "=", prefix="\n")

    # ------------------ LOAD TEXT ------------------
    print_information("Loading text...", 1, "\n")
    text = load_text(text_path)
    print_information(f"Loaded {len(text)} characters", prefix="    ")

    # -------------- LOAD CLEAN NAMES ---------------
    # # Load canonical names
    print_information("Loading canonical names...", 2, "\n")
    import pandas as pd
    clean_names = pd.read_csv(names_csv)
    print_information(f"Loaded {len(clean_names)} canonical characters", prefix="    ")

    # ------------------ LOAD MODELS ------------------
    print_information("Loading spaCy + GLiNER models...", 3, "\n")
    nlp = load_spacy_model(nlp)
    gliner_model = load_gliner_model(gliner_model)
    
    # ------------------ SEGMENT SENTENCES ------------------
    print_information("Segmenting sentences...", 4, "\n")
    sentences = segment_sentences(text, tokens_path)
    print_information(f"Found {len(sentences)} sentences", prefix="    ")

    # ------------------ RUN NER ------------------
    print_information("Running NER (GLiNER + spaCy)...", 5, "\n")
    ner_spans = run_ner(sentences, gliner_model, nlp, text)
    print_information(f"Detected {len(ner_spans)} PERSON spans", prefix="    ")

    # ------------------ BUILD VARIANT INDEX ------------------
    variant_idx = build_variant_index(clean_names)

    # ------------------ FUZZY MATCH ------------------
    print_information("Running dictionary pre-pass...", 7, "\n")
    dict_spans = fuzzy_match_ngrams(sentences, clean_names, variant_idx, text_ref=text)
    print_information(f"Found {len(dict_spans)} additional spans from dictionary", prefix="    ")

    # ------------------ DEUPLICATE SPANS ------------------
    all_spans = _deduplicate_spans(ner_spans + dict_spans)
    print_information(f"Total unique spans after merge: {len(all_spans)}", prefix="    ")

    # ------------------ BUILD ALIAS DICT ------------------
    print_information("Building alias dictionary...", 7, "\n")
    alias_dict, review_items = build_alias_dict(all_spans, clean_names, variant_idx)
    print_information(f"Auto-accepted: {len(alias_dict)} surface forms", prefix="    ")
    print_information(f"Flagged for review: {len(review_items)} surface forms", prefix="    ")

    # ------------------ PRINT STATISTICS ------------------
    if verbose:
        print_verbose_stats(all_spans, alias_dict, review_items, clean_names)

    # ------------------ SAVE RESUTS ------------------
    print_information("Saving Stage 1 outputs...", 8, "\n")
    stage_dir = out_dir / "char_resolution" if out_dir != CHAR_RES_OUT.parent else CHAR_RES_OUT
    save_ner(alias_dict, review_items, all_spans, stage_dir)
    print_information(f"Saved to → {stage_dir}", "✓", col="GREEN")

    return alias_dict, review_items


if __name__ == "__main__":
    main(verbose=True)
