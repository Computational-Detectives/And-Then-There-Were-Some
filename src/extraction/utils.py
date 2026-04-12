"""
Utility functions for the manual extraction pipeline.

All functions are COPIED from the existing codebase and adapted
to use local imports from manual_extraction.config.
Original sources are noted in docstrings.
"""
from __future__ import annotations

import re
import json
import os
import sys
import logging

from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple, Any
from colorama import Fore, Style, init

from ..config import ARTICLES, NON_NAME_WORDS, TITLES, SPACY_MODEL

init()
    

# =============================================================
# -------------------- OFFSET MANIPULATORS --------------------
# =============================================================
def build_char_offset_to_canonical(span_index: List[Dict]) -> Dict[Tuple[int, int], int]:
    """
    Build a lookup from (start_char, end_char) → canonical_id.
    """
    lookup: Dict[Tuple[int, int], int] = {}
    for span in span_index:
        if span.get("canonical_id") is not None:
            key = (span["start_char"], span["end_char"])
            lookup[key] = span["canonical_id"]
    
    return lookup


def snap_span(start: int, end: int, tokens: List[Dict], full_text: str) -> Tuple[int, int, str]:
    matched = [t for t in tokens if t["byte_offset"] > start and t["byte_onset"] < end]
    if matched:
        s = int(matched[0]["byte_onset"])
        e = int(matched[-1]["byte_offset"])
        return s, e, full_text[s:e]
    
    return start, end, full_text[start:end]


# =============================================================
# ------------------ NAME MATCHING (PRIVATE) ------------------
# =============================================================
def _normalize_name(name: str) -> str:
    """
    Normalize a name string for consistent matching.

    - Converts to lowercase
    - Removes punctuation (periods, commas, hyphens)
    - Collapses multiple spaces
    - Removes leading articles (the, a, an)
    """
    import pandas as pd
    if pd.isna(name) or not name:
        return ""

    name = name.lower()
    name = re.sub(r"[.,\"-]", "", name)
    name = " ".join(name.split())

    for article in ARTICLES:
        if name.startswith(article + " "):
            name = name[len(article):].strip()

    return name


def _clean_non_names(normalized_phrase: str, known_names: Set[str]) -> str:
    """
    Extract proper name tokens from a phrase by removing non-name words.

    For example:
    - "That fellow Lombard" -> "Lombard"
    - "Poor Dr. Armstrong" -> "Dr Armstrong"

    Copied from src/auxiliary.py.
    """
    if not normalized_phrase:
        return ""

    tokens = normalized_phrase.split()

    title_words = {"mr", "mrs", "ms", "miss", "mister", "dr", "captain", "general",
                   "inspector", "sir", "lady", "justice"}

    kept_tokens = []
    for token in tokens:
        if token in known_names or token in title_words:
            kept_tokens.append(token)
        elif token in NON_NAME_WORDS:
            continue
        elif len(token) > 2:
            kept_tokens.append(token)

    return " ".join(kept_tokens)


def _build_name_variants(row: pd.Series) -> Set[str]:
    """
    Build all possible name variants for a clean name entry.

    Generates variants from:
    - `fullname`, `firstname`, `surname`, `middlename`
    - `firstname + surname`, `firstname + middlename`
    - Aliases (from `aka` column)
    - Professions (from `profession` column)
    - `title + surname`, `title + fullname` (based on gender)

    Copied from src/extraction/match_names.py.
    """
    import pandas as pd
    variants = set()
    titles = TITLES.get(row.get("gender"), [])

    fullname = row.get("fullname")
    firstname = row.get("firstname")
    middlename = row.get("middlename")
    surname = row.get("surname")

    for field in [fullname, firstname, middlename, surname]:
        if pd.notna(field) and field:
            variants.add(_normalize_name(str(field)))

    if pd.notna(firstname) and pd.notna(surname):
        variants.add(_normalize_name(f"{firstname} {surname}"))

    if pd.notna(firstname) and pd.notna(middlename):
        variants.add(_normalize_name(f"{firstname} {middlename}"))

    aka = row.get("aka")
    if pd.notna(aka) and aka:
        for alias in str(aka).split(";"):
            variants.add(_normalize_name(alias.strip()))

    prof = row.get('profession')
    if pd.notna(prof) and prof:
        for profession in str(prof).split(";"):
            variants.add(_normalize_name(profession.strip()))

    if pd.notna(surname) and titles:
        for title in titles:
            variants.add(_normalize_name(f"{title} {surname}"))
            if pd.notna(fullname):
                variants.add(_normalize_name(f"{title} {fullname}"))

    variants.discard("")
    return variants


# =============================================================
# ------------------ NAME MATCHING (PUBLIC) -------------------
# =============================================================
def build_variant_index(
    names_df: pd.DataFrame,
) -> Tuple[Dict[str, List[int]], List[str], Dict[int, str], Set[str]]:
    """
    Build lookup structures for efficient matching.

    Returns:
        - variant_to_ids: Dict mapping each variant string -> list of possible IDs
        - all_variants: Flat list of all variants (for rapidfuzz)
        - id_to_gender: Dict mapping ID -> gender ('m', 'f')
        - all_name_tokens: Set of all individual name tokens

    Copied from src/extraction/match_names.py.
    """
    variant_to_ids: Dict[str, List[int]] = {}
    id_to_gender: Dict[int, str] = {}

    for _, row in names_df.iterrows():
        row_id = int(row["id"])
        id_to_gender[row_id] = row["gender"]

        for variant in _build_name_variants(row):
            if variant not in variant_to_ids:
                variant_to_ids[variant] = []
            if row_id not in variant_to_ids[variant]:
                variant_to_ids[variant].append(row_id)

    all_variants = list(variant_to_ids.keys())

    all_name_tokens = set()
    for variant in all_variants:
        all_name_tokens.update(variant.split())

    return variant_to_ids, all_variants, id_to_gender, all_name_tokens


def match_name(
    name: str,
    gender: str,
    variant_to_ids: Dict[str, List[int]],
    all_variants: List[str],
    id_to_gender: Dict[int, str],
    names_df: pd.DataFrame,
    all_name_tokens: Set[str],
    threshold: float = 60.0,
) -> Tuple[Optional[int], str, float, Optional[str]]:
    """
    Match a single name to the best candidate in the database.

    Uses rapidfuzz's token_sort_ratio which is robust against word reordering
    and handles partial matches well.

    Copied from src/extraction/match_names.py.
    """
    original_name = name
    normalized = _normalize_name(name)
    cleaned = _clean_non_names(normalized, all_name_tokens)

    if not cleaned:
        return None, original_name, 0.0, None
        
    import pandas as pd
    from rapidfuzz import process, fuzz

    def try_fuzzy_match(query: str) -> Tuple[Optional[int], Optional[str], float, Optional[str]]:
        matches = process.extract(
            query, all_variants, scorer=fuzz.token_sort_ratio, limit=10
        )

        for variant, score, _ in matches:
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

    for enforce_gender in [True]:
        cid, fullname, score, corrected_gender = try_fuzzy_match(cleaned)

        if cid is not None:
            penalty = 1.0 if enforce_gender else 0.8
            return cid, fullname, score * penalty, corrected_gender

        tokens = cleaned.split()
        for token in tokens:
            if token in all_name_tokens and len(token) > 2:
                cid, fullname, score, corrected_gender = try_fuzzy_match(token)
                if cid is not None:
                    penalty = 0.9 if enforce_gender else 0.7
                    return cid, fullname, score * penalty, corrected_gender

    return None, original_name, 0.0, None


# =============================================================
# ---------------------- TOKEN COMPONENTS ---------------------
# =============================================================
def is_negated(verb_component) -> bool:
    """
    Check if the verb is negated by looking for a child with dep_="neg".

    Examples of negation: "does not like", "didn't see", "never went"

    Copied from src/extraction/extract_svo_triples.py.
    """
    from spacy.tokens import Span

    if isinstance(verb_component, list):
        tokens = verb_component
    elif isinstance(verb_component, Span):
        tokens = list(verb_component)
    else:
        tokens = [verb_component]

    for token in tokens:
        for child in token.children:
            if child.dep_ == "neg":
                return True
    return False


def get_compound_tokens(token) -> list:
    """
    Get all tokens that form a compound noun with the given token.
    Returns tokens in document order.
    """
    compound_tokens = [token]

    for child in token.children:
        if child.dep_ == "compound":
            compound_tokens.extend(get_compound_tokens(child))

    compound_tokens.sort(key=lambda t: t.i)
    return compound_tokens


def get_noun_info(component) -> tuple:
    """
    Extract info for subject/object components, expanding compound nouns.

    Copied from src/extraction/extract_svo_triples.py.
    """
    from spacy.tokens import Span

    if isinstance(component, list):
        base_tokens = component
    elif isinstance(component, Span):
        base_tokens = list(component)
    else:
        base_tokens = [component]

    all_tokens = []
    seen = set()
    for token in base_tokens:
        for t in get_compound_tokens(token):
            if t.i not in seen:
                all_tokens.append(t)
                seen.add(t.i)

    all_tokens.sort(key=lambda t: t.i)

    if not all_tokens:
        return "", "", [], None, "NOUN"

    ids = [t.i for t in all_tokens]
    text = " ".join([t.text for t in all_tokens])
    lemma = " ".join([t.lemma_ for t in all_tokens])
    primary_id = ids[0]

    if isinstance(component, Span):
        pos = component.root.pos_
    elif isinstance(component, list):
        pos = component[-1].pos_ if component else "NOUN"
    else:
        pos = component.pos_

    return text, lemma, ids, primary_id, pos


def get_verb_info(component) -> tuple:
    """
    Extract info for verb components, keeping only the main verb (not auxiliaries).

    Copied from src/extraction/extract_svo_triples.py.
    """
    from spacy.tokens import Span

    if isinstance(component, list):
        tokens = component
    elif isinstance(component, Span):
        tokens = list(component)
    else:
        tokens = [component]

    main_verbs = [t for t in tokens if t.pos_ == "VERB" and t.dep_ not in ("aux", "auxpass")]

    if not main_verbs:
        main_verbs = [t for t in tokens if t.dep_ not in ("aux", "auxpass")]

    if not main_verbs:
        main_verbs = tokens

    main_verbs.sort(key=lambda t: t.i)

    if not main_verbs:
        return "", "", [], None, "VERB"

    ids = [t.i for t in main_verbs]
    text = " ".join([t.text for t in main_verbs])
    lemma = " ".join([t.lemma_ for t in main_verbs])
    primary_id = ids[0]
    pos = main_verbs[0].pos_

    return text, lemma, ids, primary_id, pos


# =============================================================
# ------------------- SENTENCE RECONSTRUCTION -----------------
# =============================================================
def make_doc_from_sentence(sentence: pd.DataFrame, nlp) -> spacy.tokens.Doc:
    """
    Reconstructs a spaCy Doc from a sentence DataFrame.
    """
    import numpy as np
    import pandas as pd
    from spacy.tokens import Doc, Token
    from spacy.attrs import HEAD, DEP

    if not Token.has_extension("global_id"):
        Token.set_extension("global_id", default=None)

    # Extract all words in the sentence & compute location of spaces
    words = sentence["word"].tolist()
    spaces = [True] * (len(words) - 1) + [False]
    
    # Create custom spacy.Doc object for the current sentence
    doc = Doc(nlp.vocab, words=words, spaces=spaces)

    # Compute sentence-local token IDs. Required by spacy.Doc()
    global_to_local = {row["token_ID_within_document"]: idx
                       for idx, (_, row) in enumerate(sentence.iterrows())}

    heads = []
    deps  = []

    # Add token indices relative to sentence head & add dependencies to the vocabulary
    for idx, (_, row) in enumerate(sentence.iterrows()):
        global_head_idx = int(row["syntactic_head_ID"])
        local_head_idx = global_to_local.get(global_head_idx, idx)  # Fallback to self if missing
        rel_head = local_head_idx - idx

        heads.append(rel_head)
        deps.append(nlp.vocab.strings.add(str(row["dependency_relation"])))

    n = len(heads)
    arr = np.zeros((n, 2), dtype="uint64")

    for i in range(n):
        arr[i, 0] = np.int64(heads[i]).view(np.uint64)
        arr[i, 1] = deps[i]

    # Add head and dependency information to Doc-object
    doc.from_array([HEAD, DEP], arr)

    # Assign POS / TAG / LEMMA and store global token IDs
    for token, (_, row) in zip(doc, sentence.iterrows()):
        token.pos_   = str(row["POS_tag"])
        token.tag_   = str(row["fine_POS_tag"])
        token.lemma_ = str(row["lemma"])
        # Store global token ID as custom attribute
        token._.global_id = row["token_ID_within_document"]

    return doc


# =============================================================
# ----------------------- MODEL LOADING -----------------------
# =============================================================

class SpacyServerProxy:
    """
    Drop-in replacement for a spaCy Language object that delegates
    inference to a running model_server.py and returns real
    spacy.tokens.Doc objects.

    Stages 3-4 only use nlp.vocab (via make_doc_from_sentence),
    which is satisfied by a blank English vocab.
    Stages 0-1 call nlp(text) which this proxy handles via HTTP.
    """

    def __init__(self, server_url: str):
        import spacy
        self.server_url = server_url.rstrip("/")
        self._blank = spacy.blank("en")
        self.vocab = self._blank.vocab
        # Expose Defaults so tokeniser customisation code doesn't crash
        self.Defaults = self._blank.Defaults

    def __call__(self, text: str):
        """POST text to /process and reconstruct a spaCy Doc."""
        import requests
        import numpy as np
        from spacy.tokens import Doc
        from spacy.attrs import HEAD, DEP

        resp = requests.post(
            f"{self.server_url}/process",
            json={"text": text},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        all_toks = data["tokens"]
        if not all_toks:
            return Doc(self.vocab)

        # Build word list and spaces
        words = [t["text"] for t in all_toks]
        spaces = [bool(t.get("whitespace", "")) for t in all_toks]

        doc = Doc(self.vocab, words=words, spaces=spaces)

        # Reconstruct sentence boundaries BEFORE dependencies to avoid E043
        n = len(all_toks)
        sent_starts = [False] * n
        for s in data.get("sentences", []):
            if s["start"] < n:
                sent_starts[s["start"]] = True
        if sent_starts and not any(sent_starts):
            sent_starts[0] = True  # ensure at least one sentence
        for token, is_start in zip(doc, sent_starts):
            token.is_sent_start = is_start

        # Set per-token attributes
        for token, t in zip(doc, all_toks):
            token.pos_ = t["pos"]
            token.tag_ = t["tag"]
            token.lemma_ = t["lemma"]

        # Set HEAD + DEP via numpy array
        arr = np.zeros((n, 2), dtype="uint64")
        for i, t in enumerate(all_toks):
            rel_head = t["head_i"] - t["i"]
            arr[i, 0] = np.int64(rel_head).view(np.uint64)
            arr[i, 1] = self.vocab.strings.add(t["dep"])
        doc.from_array([HEAD, DEP], arr)

        # Reconstruct NER entities
        ents = []
        for e in data.get("entities", []):
            span = doc.char_span(
                e["start_char"], e["end_char"], label=e["label"]
            )
            if span is not None:
                ents.append(span)
        doc.ents = ents

        return doc


def load_spacy_model(nlp=None, server_url=None):
    """
    Load the spaCy model.

    If ``server_url`` is given, returns a :class:`SpacyServerProxy`
    that delegates inference to the model server (zero local GPU/RAM).
    """
    if server_url is not None:
        print_information(
            f"Using model server at {server_url}", prefix="    "
        )
        return SpacyServerProxy(server_url)


    if nlp is None:
        import spacy
        from spacy.util import compile_infix_regex
        nlp = spacy.load(SPACY_MODEL)

        infixes = list(nlp.Defaults.infixes) + [r'(?<=[–—])[–—]', r'[–—]']
        nlp.tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
        print_information("Model loaded", prefix="    ")
    else:
        print_information("Using pre-loaded global spaCy model.", prefix="    ")

    return nlp


def load_gliner_model(gliner_model=None):
    """Load the provided GLiNER model"""
    if gliner_model is None:
        from gliner import GLiNER
        gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        print_information("GLiNER model loaded", prefix="    ")
    else:
        print_information("Using pre-loaded global GLiNER model", prefix="    ")

    return gliner_model


# =============================================================
# ------------------------ MISC LOADING -----------------------
# =============================================================
def load_span_index(path: Path) -> list[dict]:
    """Load span_index.jsonl from Stage 2."""
    spans = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            spans.append(json.loads(line))
    return spans


def load_alias_dict(path: Path) -> Dict[Any]:
    """Load alias_dict.json from Stage 1 output."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: Path) -> str:
    """Read the raw text file and normalise whitespace depending on extension."""
    if path.suffix.lower() == ".txt":
        print(f"    Preprocessing .txt file: {path.name}")
        # from src.auxiliary import preprocess
        # preproc_path = out_dir / "preproc_attwn.txt"
        #  preprocess(str(path), str(preproc_path))
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        return text
    else:
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

        # Remove leading/trailing whitespace around '\n'
        return re.sub(r"[^\S\n]*\n[^\S\n]*", "\n", text)
    

# =============================================================
# ---------------------- PRINT SUPPRESSOR ---------------------
# =============================================================
@contextmanager
def suppress_stdout():
    """
    Redirect Python print statements from external libraries to '/dev/null'
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# =============================================================
# ---------------------- PRINTING HELPERS ---------------------
# =============================================================
logger = logging.getLogger("pipeline")

def setup_pipeline_logger(log_path: Path | str) -> logging.Logger:
    """Call once at the start of each script (main and child)."""
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return logger


def log_print(*args, **kwargs):
    """Drop-in ``print()`` replacement that also writes to the pipeline logger."""
    msg = " ".join(str(a) for a in args)
    print(msg)
    logger.info(msg)


def print_headers(msg: str, symb: str, prefix: str = ""):
    line = f'{prefix}{symb * 80}\n{msg}\n{symb * 80}'
    print(line)
    logger.info(line)


def print_information(msg: str, symb: Optional[str | int] = None, prefix: str = "", col: str = "BLUE"):
    if symb:
        colour = getattr(Fore, col)
        print(f'{prefix}[{colour}{symb}{Style.RESET_ALL}] {msg}')
        logger.info(f'{prefix}[{symb}] {msg}')
    else:
        print(f'{prefix}{msg}')
        logger.info(f'{prefix}{msg}')