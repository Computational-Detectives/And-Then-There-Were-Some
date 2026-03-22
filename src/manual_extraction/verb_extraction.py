"""
Stage 3 — Verb/Action Extraction

For every resolved character span, find the verb where that character
is the grammatical subject, lemmatise it, flag negation, and classify
the verb via VerbNet/FrameNet.
"""
from __future__ import annotations

import spacy
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from .config import COREF_OUT, TOKENS, VERB_OUT, CLEAN_NAMES
from .utils import (
    is_negated, print_headers, print_information, 
    load_spacy_model, load_span_index, 
    build_char_offset_to_canonical,
    make_doc_from_sentence
    )


# ============================================================================
# VERBNET / FRAMENET CATEGORISATION
# ============================================================================

# Top-level VerbNet class roots → readable category labels
VERBNET_CATEGORIES = {
    "say": "COMMUNICATION",
    "tell": "COMMUNICATION",
    "talk": "COMMUNICATION",
    "ask": "COMMUNICATION",
    "call": "COMMUNICATION",
    "whisper": "COMMUNICATION",
    "shout": "COMMUNICATION",
    "murmur": "COMMUNICATION",
    "speak": "COMMUNICATION",
    "reply": "COMMUNICATION",
    "answer": "COMMUNICATION",
    "declare": "COMMUNICATION",
    "announce": "COMMUNICATION",
    "explain": "COMMUNICATION",
    "confess": "COMMUNICATION",
    "murder": "KILLING",
    "kill": "KILLING",
    "die": "DEATH",
    "drown": "DEATH",
    "run": "MOTION",
    "walk": "MOTION",
    "go": "MOTION",
    "come": "MOTION",
    "move": "MOTION",
    "turn": "MOTION",
    "enter": "MOTION",
    "leave": "MOTION",
    "cross": "MOTION",
    "follow": "MOTION",
    "see": "PERCEPTION",
    "look": "PERCEPTION",
    "watch": "PERCEPTION",
    "hear": "PERCEPTION",
    "notice": "PERCEPTION",
    "stare": "PERCEPTION",
    "observe": "PERCEPTION",
    "feel": "PERCEPTION",
    "think": "COGNITION",
    "know": "COGNITION",
    "remember": "COGNITION",
    "believe": "COGNITION",
    "realise": "COGNITION",
    "realize": "COGNITION",
    "understand": "COGNITION",
    "wonder": "COGNITION",
    "suppose": "COGNITION",
    "suspect": "COGNITION",
    "consider": "COGNITION",
    "get": "OBTAINING",
    "take": "OBTAINING",
    "find": "OBTAINING",
    "give": "TRANSFER",
    "hand": "TRANSFER",
    "send": "TRANSFER",
    "bring": "TRANSFER",
    "pass": "TRANSFER",
    "hit": "CONTACT",
    "strike": "CONTACT",
    "touch": "CONTACT",
    "push": "CONTACT",
    "pull": "CONTACT",
    "put": "PLACEMENT",
    "set": "PLACEMENT",
    "place": "PLACEMENT",
    "lay": "PLACEMENT",
    "sit": "BODY_POSITION",
    "stand": "BODY_POSITION",
    "lie": "BODY_POSITION",
    "rise": "BODY_POSITION",
    "eat": "CONSUMPTION",
    "drink": "CONSUMPTION",
    "want": "DESIRE",
    "wish": "DESIRE",
    "hope": "DESIRE",
    "try": "ATTEMPT",
    "attempt": "ATTEMPT",
    "begin": "ASPECT",
    "start": "ASPECT",
    "stop": "ASPECT",
    "continue": "ASPECT",
    "finish": "ASPECT",
    "like": "EMOTION",
    "love": "EMOTION",
    "hate": "EMOTION",
    "fear": "EMOTION",
    "enjoy": "EMOTION",
    "be": "STATE",
    "have": "POSSESSION",
    "own": "POSSESSION",
    "keep": "POSSESSION",
    "hold": "POSSESSION",
    "make": "CREATION",
    "create": "CREATION",
    "build": "CREATION",
    "help": "SOCIAL",
    "meet": "SOCIAL",
    "join": "SOCIAL",
    "agree": "SOCIAL",
    "refuse": "SOCIAL",
    "deny": "SOCIAL",
    "accuse": "SOCIAL",
    "blame": "SOCIAL",
    "thank": "SOCIAL",
    "trust": "SOCIAL",
}


def print_verbose_statistics(verbs_df: pd.DataFrame, n_passive: int, n_negated: int) -> None:
    print_headers("STAGE 3 — VERBOSE STATISTICS", "-", prefix="\n")

    # Per-character verb counts
    id_to_name = {}
    try:
        names_df = pd.read_csv(CLEAN_NAMES)
        id_to_name = dict(zip(names_df["id"], names_df["fullname"]))
    except Exception:
        pass

    char_verb_counts = verbs_df.groupby("canonical_id").size().sort_values(ascending=False)
    print("    Per-Character Verb Counts:")
    for cid, count in char_verb_counts.items():
        name = id_to_name.get(int(cid), f"char_{cid}")
        print(f"      {name:35s} {count:5d} verbs")

    # Passive & negated breakdown
    print(f"\n    Verb Modifiers:")
    print(f"      Passive constructions: {n_passive:5d}  ({n_passive / len(verbs_df) * 100:.1f}%)")
    print(f"      Negated verbs:         {n_negated:5d}  ({n_negated / len(verbs_df) * 100:.1f}%)")

    # Top verb categories
    print("\n    Top Verb Categories:")
    cat_counts = verbs_df["verb_category"].value_counts()
    for cat, count in cat_counts.head(15).items():
        pct = count / len(verbs_df) * 100
        label = cat if pd.notna(cat) else "(uncategorised)"
        print(f"      {label:20s} {count:5d}  ({pct:5.1f}%)")

    # Top verb lemmas overall
    print("\n    Top 15 Verb Lemmas:")
    lemma_counts = verbs_df["verb_lemma"].value_counts()
    for lemma, count in lemma_counts.head(15).items():
        print(f"      {lemma:20s} {count:5d}")


def classify_verb(lemma: str) -> Optional[str]:
    """
    Classify a verb lemma using VerbNet/FrameNet, with a hardcoded
    fallback mapping for common verbs.

    :param lemma: The lemmatised verb.
    :return: Category string or None if not found.
    """
    lemma_lower = lemma.lower().strip()

    # Hardcoded lookup first (fastest)
    if lemma_lower in VERBNET_CATEGORIES:
        return VERBNET_CATEGORIES[lemma_lower]

    # Try NLTK VerbNet
    try:
        from nltk.corpus import verbnet as vn
        classids = vn.classids(lemma_lower)
        if classids:
            # Take the first class name root
            class_name = classids[0].split("-")[0]
            return VERBNET_CATEGORIES.get(class_name, class_name.upper())
    except (LookupError, ImportError):
        pass

    # Try NLTK FrameNet
    try:
        from nltk.corpus import framenet as fn
        frames = fn.frames_by_lemma(rf"(?i){lemma_lower}")
        if frames:
            return frames[0].name.upper()
    except (LookupError, ImportError):
        pass

    return None


# ============================================================================
# VERB EXTRACTION
# ============================================================================

def extract_verbs_for_sentence(
    doc: spacy.tokens.Doc,
    char_lookup: Dict[Tuple[int, int], int],
    sentence_df: pd.DataFrame,
) -> list[dict]:
    """
    Extract (character, verb) pairs from a spaCy Doc sentence.

    :param doc: Sentence-level spaCy Doc.
    :param char_lookup: (start_char, end_char) → canonical_id.
    :param sid: Sentence ID.
    :param sentence_df: DataFrame chunk for this sentence.
    :return: List of verb dicts.
    """
    results: List[Dict] = []

    for token in doc:
        # Check if this token is part of a resolved character span
        # Get row of the token within sentence_df
        tok_row = sentence_df.iloc[token.i]
        token_start = int(tok_row["byte_onset"])
        token_end = int(tok_row["byte_offset"])

        # Find matching character span
        matched_cid = None
        for (s, e), cid in char_lookup.items():
            if s <= token_start and token_end <= e:
                matched_cid = cid
                break

        if matched_cid is None:
            continue

        # Check if this token is an nsubj or nsubjpass of a verb
        if token.dep_ in ("nsubj", "nsubjpass"):
            verb = token.head
            if verb.pos_ not in ("VERB", "AUX"):
                continue

            is_passive = token.dep_ == "nsubjpass"
            negated = is_negated(verb)

            # Light verb handling: check for xcomp/ccomp
            verb_text = verb.text
            verb_lemma = verb.lemma_
            for child in verb.children:
                if child.dep_ in ("xcomp", "ccomp") and child.pos_ == "VERB":
                    verb_text = f"{verb.text} {child.text}"
                    verb_lemma = f"{verb.lemma_} {child.lemma_}"
                    break
            
            # Get row of the verb found
            verb_tok_row = sentence_df.iloc[verb.i]            

            results.append({
                "canonical_id": matched_cid,
                "verb_text": verb_text,
                "verb_lemma": verb_lemma,
                "negated": negated,
                "is_passive": is_passive,
                "sid": verb_tok_row['sentence_ID'],
                "verb_token_idx": verb._.global_id,
            })

            # Handle conjoined verbs sharing the same subject
            for child in verb.children:
                if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
                    conj_negated = is_negated(child)
                    conj_text = child.text
                    conj_lemma = child.lemma_
                    conj_tok_row = sentence_df.iloc[child.i]

                    results.append({
                        "canonical_id": matched_cid,
                        "verb_text": conj_text,
                        "verb_lemma": conj_lemma,
                        "negated": conj_negated,
                        "is_passive": is_passive,
                        "sid": conj_tok_row['sentence_ID'],
                        "verb_token_idx": child._.global_id,
                    })

    return results


def extract_all_verbs(
    tokens_df: pd.DataFrame,
    span_index: list[dict],
    nlp: spacy.language.Language,
) -> pd.DataFrame:
    """
    Process all sentences and extract (character, verb) pairs.

    :return: DataFrame with columns: canonical_id, verb_text, verb_lemma,
             negated, is_passive, sid, verb_token_idx, verb_category.
    """
    from .utils import make_doc_from_sentence

    all_verbs: list[dict] = []
    char_lookup = build_char_offset_to_canonical(span_index)
    for _, sentence_df in tqdm(tokens_df.groupby("sentence_ID")):
        # Create spaCy.Doc object from current sentence
        doc = make_doc_from_sentence(sentence_df, nlp)

        # Extract the verbs in the current sentence
        verbs = extract_verbs_for_sentence(doc, char_lookup, sentence_df)
        all_verbs.extend(verbs)

    df = pd.DataFrame(all_verbs)

    # Categorise verbs
    if not df.empty:
        # Add verb category
        df["verb_category"] = df["verb_lemma"].apply(
            lambda l: classify_verb(l.split()[0])  # classify main verb in light-verb constructions
        )

    return df


# ============================================================================
# SAVE
# ============================================================================

def save_stage3(verbs_df: pd.DataFrame, stage_dir: Path = VERB_OUT) -> None:
    """Save the character_verbs.csv file."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    verbs_df.to_csv(stage_dir / "character_verbs.csv", sep="\t", index=False)


# ============================================================================
# MAIN
# ============================================================================

def main(
    tokens_path: Path = TOKENS,
    coref_dir: Path = COREF_OUT,
    out_dir: Path = VERB_OUT,
    verbose: bool = False,
    nlp: spacy.language.Language = None,
) -> pd.DataFrame:
    """Run Stage 3: verb/action extraction with VerbNet classification."""

    print_headers("STAGE 3 — VERB/ACTION EXTRACTION", "=", prefix="\n")

    # Load inputs
    print_information("Loading inputs...", 1, "\n")
    tokens_df = pd.read_csv(tokens_path, sep="\t", keep_default_na=False)
    span_index = load_span_index(coref_dir / "span_index.jsonl")
    print_information(f"Loaded {len(span_index)} resolved spans", prefix="    ")
    print_information(f"Loaded {len(tokens_df)} tokens", prefix="    ")

    # Load spaCy
    nlp = load_spacy_model(nlp)

    # Extract verbs
    print_information("Extracting verbs...", 3, "\n")
    verbs_df = extract_all_verbs(tokens_df, span_index, nlp)
    print_information(f"Extracted {len(verbs_df)} (character, verb) pairs", prefix="    ")

    if not verbs_df.empty:
        categorised = verbs_df["verb_category"].notna().sum()
        uncategorised = len(verbs_df) - categorised
        n_unique_verbs = verbs_df["verb_lemma"].nunique()
        n_unique_chars = verbs_df["canonical_id"].nunique()
        n_passive = verbs_df["is_passive"].sum() if "is_passive" in verbs_df.columns else 0
        n_negated = verbs_df["negated"].sum() if "negated" in verbs_df.columns else 0

        print_information(f"VerbNet/FrameNet categorised: {categorised}/{len(verbs_df)} "
                          f"({categorised/len(verbs_df)*100:.1f}%)", prefix="    ")
        print_information(f"Uncategorised: {uncategorised}", prefix="    ")
        print_information(f"Unique verb lemmas: {n_unique_verbs}", prefix="    ")
        print_information(f"Characters with verbs: {n_unique_chars}", prefix="    ")

        if verbose:
            print_verbose_statistics(verbs_df, n_passive, n_negated)

    # Save
    print_information("Saving Stage 3 outputs...", 4, "\n")
    stage_dir = out_dir / "verb_extraction"
    save_stage3(verbs_df, stage_dir)
    print_information(f"Saved to → {stage_dir}", "✓", col="GREEN")

    return verbs_df


if __name__ == "__main__":
    main(verbose=True)
