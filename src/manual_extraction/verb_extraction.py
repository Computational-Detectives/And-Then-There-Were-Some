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
def split_on_quotes(sentence_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Recursively split a sentence DataFrame on quoted spans,
    returning a flat list of DataFrames each representing
    a parseable unit, preserving original byte offsets throughout.
    """
    # Find quote token indices within this df
    words = sentence_df["word"].tolist()
    open_chars  = set(['"', '\u201c', '\u2018'])
    close_chars = set(['"', '\u201d', '\u2019'])

    open_idx = next(
        (i for i, w in enumerate(words) if w in open_chars), None
    )
    if open_idx is None:
        # Base case: no quotes found, return as-is
        return [sentence_df]

    # Find matching closing quote
    close_idx = next(
        (i for i in range(open_idx + 1, len(words)) if words[i] in close_chars),
        None
    )
    if close_idx is None:
        # Unmatched quote, return as-is
        return [sentence_df]

    # Split into three parts:
    # 1. Main clause before the quote (exclusive of quote marks)
    # 2. Quoted content (exclusive of quote marks)
    # 3. Remainder after the closing quote mark
    before = sentence_df.iloc[:open_idx]
    quoted = sentence_df.iloc[open_idx + 1 : close_idx]
    after  = sentence_df.iloc[close_idx + 1:]

    # Combine before+after as the main clause
    main = pd.concat([before, after]).reset_index(drop=True)

    result = []
    # Recurse into main clause (may contain further quotes)
    if not main.empty:
        result.extend(split_on_quotes(main))
    # Recurse into quoted span (may itself contain nested quotes)
    if not quoted.empty:
        result.extend(split_on_quotes(quoted))

    return result


def extract_verbs_for_sentence_orig(
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

def extract_verbs_for_sentence_rec(
    doc: spacy.tokens.Doc,
    char_lookup: Dict[Tuple[int, int], int],
    sentence_df: pd.DataFrame,
) -> list[dict]:
    results: List[Dict] = []

    def extract_from_verb(verb, visited: set) -> None:
        """Recursively extract (subject, verb) pairs from a verb and its ccomp children."""
        if verb.i in visited:
            return
        visited.add(verb.i)

        # Find the subject of this specific verb
        subject = next(
            (c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")),
            None
        )

        if subject is not None:
            tok_row = sentence_df.iloc[subject.i]
            token_start = int(tok_row["byte_onset"])
            token_end   = int(tok_row["byte_offset"])

            matched_cid = next(
                (cid for (s, e), cid in char_lookup.items()
                 if s <= token_start and token_end <= e),
                None
            )

            if matched_cid is not None:
                is_passive = subject.dep_ == "nsubjpass"
                negated    = is_negated(verb)

                # xcomp only (not ccomp) for light verb constructions
                verb_text  = verb.text
                verb_lemma = verb.lemma_
                for child in verb.children:
                    if child.dep_ == "xcomp" and child.pos_ == "VERB":
                        verb_text  = f"{verb.text} {child.text}"
                        verb_lemma = f"{verb.lemma_} {child.lemma_}"
                        break

                verb_tok_row = sentence_df.iloc[verb.i]
                results.append({
                    "canonical_id":    matched_cid,
                    "verb_text":       verb_text,
                    "verb_lemma":      verb_lemma,
                    "negated":         negated,
                    "is_passive":      is_passive,
                    "sid":             verb_tok_row["sentence_ID"],
                    "verb_token_idx":  verb._.global_id,
                })

                # Handle conjoined verbs
                for child in verb.children:
                    if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
                        extract_from_verb(child, visited)

        # Recurse into ccomp children regardless of whether this verb
        # had a matched subject — the embedded clause may have its own
        for child in verb.children:
            if child.dep_ == "ccomp" and child.pos_ in ("VERB", "AUX"):
                extract_from_verb(child, visited)

    # Entry point: find the ROOT and recurse from there
    for token in doc:
        if token.dep_ == "ROOT":
            extract_from_verb(token, visited=set())
            break

    return results


def get_full_xcomp_chain(verb) -> tuple[str, str]:
    """
    Follow xcomp chain downward from a verb, returning the
    concatenated text and lemma of the full aspectual construction.
    e.g. "imagine wanting die" / "imagine want die"
    """
    texts  = [verb.text]
    lemmas = [verb.lemma_]
    current = verb
    while True:
        xcomp_child = next(
            (c for c in current.children
             if c.dep_ == "xcomp" and c.pos_ == "VERB"),
            None
        )
        if xcomp_child is None:
            break
        texts.append(xcomp_child.text)
        lemmas.append(xcomp_child.lemma_)
        current = xcomp_child
    return " ".join(texts), " ".join(lemmas)


def extract_verbs_for_sentence(
    doc: spacy.tokens.Doc,
    char_lookup: Dict[Tuple[int, int], int],
    sentence_df: pd.DataFrame,
) -> list[dict]:
    results: List[Dict] = []
    visited_verbs: set = set()

    def record_verb(verb, matched_cid, is_passive):
        """Record a verb and follow its xcomp chain and conj children."""
        if verb.i in visited_verbs:
            return
        visited_verbs.add(verb.i)

        negated = is_negated(verb)
        verb_text, verb_lemma = get_full_xcomp_chain(verb)
        verb_tok_row = sentence_df.iloc[verb.i]

        results.append({
            "canonical_id":   matched_cid,
            "verb_text":      verb_text,
            "verb_lemma":     verb_lemma,
            "negated":        negated,
            "is_passive":     is_passive,
            "sid":            verb_tok_row["sentence_ID"],
            "verb_token_idx": verb._.global_id,
        })

        # Conjoined verbs share the same subject
        for child in verb.children:
            if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
                record_verb(child, matched_cid, is_passive)

    def follow_ccomp(verb):
        """Recursively follow ccomp children, extracting their own subjects."""
        for child in verb.children:
            if child.dep_ != "ccomp" or child.pos_ not in ("VERB", "AUX"):
                continue
            # Check if the ccomp verb has its own subject
            subject = next(
                (c for c in child.children 
                 if c.dep_ in ("nsubj", "nsubjpass")),
                None
            )
            if subject is not None:
                tok_row = sentence_df.iloc[subject.i]
                token_start = int(tok_row["byte_onset"])
                token_end   = int(tok_row["byte_offset"])
                matched_cid = next(
                    (cid for (s, e), cid in char_lookup.items()
                     if s <= token_start and token_end <= e),
                    None
                )
                if matched_cid is not None:
                    record_verb(child, matched_cid, subject.dep_ == "nsubjpass")
            # Recurse regardless — embedded clause may have further ccomp
            follow_ccomp(child)

    # Primary pass: flat iteration over all tokens, same as original
    for token in doc:
        if token.dep_ not in ("nsubj", "nsubjpass"):
            continue

        verb = token.head
        if verb.pos_ not in ("VERB", "AUX"):
            continue

        tok_row = sentence_df.iloc[token.i]
        token_start = int(tok_row["byte_onset"])
        token_end   = int(tok_row["byte_offset"])

        matched_cid = next(
            (cid for (s, e), cid in char_lookup.items()
             if s <= token_start and token_end <= e),
            None
        )

        if matched_cid is None:
            continue

        is_passive = token.dep_ == "nsubjpass"
        record_verb(verb, matched_cid, is_passive)

        # Secondary pass: follow ccomp chains from this verb
        follow_ccomp(verb)

    return results


def extract_all_verbs(
    tokens_df: pd.DataFrame,
    span_index: list[dict],
    nlp: spacy.language.Language,
    comp: bool=False
) -> pd.DataFrame:
    """
    Process all sentences and extract (character, verb) pairs.

    :return: DataFrame with columns: canonical_id, verb_text, verb_lemma,
             negated, is_passive, sid, verb_token_idx, verb_category.
    """
    all_verbs: list[dict] = []
    char_lookup = build_char_offset_to_canonical(span_index)
    span_ranges = list(char_lookup.keys())
    for _, sentence_df in tqdm(tokens_df.groupby("sentence_ID")):
        # # Create spaCy.Doc object from current sentence
        # if not comp:
        #     doc = make_doc_from_sentence(sentence_df, nlp)
        #     # Extract the verbs in the current sentence
        #     verbs = extract_verbs_for_sentence(doc, char_lookup, sentence_df)
        #     all_verbs.extend(verbs)
        # else:

        # Check if any token in this sentence falls within a character span
        sent_onset = sentence_df["byte_onset"].min()
        sent_offset = sentence_df["byte_offset"].max()

        has_span = any(
            s <= sent_offset and e >= sent_onset
            for s, e in span_ranges
        )

        if not has_span:
            continue
        for sub_df in split_on_quotes(sentence_df):
            doc = make_doc_from_sentence(sub_df, nlp)
            verbs = extract_verbs_for_sentence(doc, char_lookup, sub_df)
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
    print_information("Loading spaCy model...", 2, "\n")
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
