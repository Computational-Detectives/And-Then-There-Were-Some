"""
Stage 4 — Object Extraction & Final Triple Assembly

Extract grammatical objects for each (character, verb) pair, classify
objects as protagonist or entity, and assemble the final triples
in the target TSV schema.
"""
from __future__ import annotations

import spacy
import pandas as pd

from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple, Set

from .config import RAW_TEXT, CLEAN_NAMES, OUT_DIR, TOKENS, COREF_OUT, VERB_OUT, OBJ_OUT, PRONOUNS
from .utils import (
    get_compound_tokens, get_noun_info, is_negated,
    print_headers, print_information, load_span_index,
    load_alias_dict, load_spacy_model, load_gliner_model,
    build_char_offset_to_canonical, make_doc_from_sentence,
    load_text
)

def print_verbose_stats(protagonist_df: pd.DataFrame, non_protagonist_df: Optional[pd.DataFrame], include_non_protagonist: Optional[bool] = False):
    print_headers("STAGE 4 — VERBOSE STATISTICS", "-", prefix="\n")

    if not protagonist_df.empty:
        # Unique interaction pairs
        pairs = protagonist_df.groupby(["name_left", "name_right"]).size().sort_values(ascending=False)
        print(f"    Unique Character Interaction Pairs: {len(pairs)}")
        print("\n    Top 15 Interaction Pairs:")
        for (left, right), count in pairs.head(15).items():
            print(f"      {left:20s} → {right:20s}  {count:4d} triples")

        # Per-character triple counts (as agent)
        agent_counts = protagonist_df["name_left"].value_counts()
        print("\n    Per-Character Triple Count (as agent):")
        for name, count in agent_counts.items():
            print(f"      {name:35s} {count:5d}")

        # Per-character triple counts (as patient)
        patient_counts = protagonist_df["name_right"].value_counts()
        print("\n    Per-Character Triple Count (as patient):")
        for name, count in patient_counts.items():
            print(f"      {name:35s} {count:5d}")

        # Negation stats
        n_negated = protagonist_df["negated"].sum()
        print(f"\n    Negated triples: {n_negated} / {len(protagonist_df)} "
                f"({n_negated/len(protagonist_df)*100:.1f}%)")

        # Top verbs in triples
        top_verbs = protagonist_df["lemma"].value_counts().head(15)
        print("\n    Top 15 Verbs in Protagonist Triples:")
        for verb, count in top_verbs.items():
            print(f"      {verb:20s} {count:5d}")

        # Sample triples
        print_headers("SAMPLE TRIPLES (first 10)", "-", prefix="\n")
        display_cols = ["name_left", "word", "name_right", "negated"]
        available_cols = [c for c in display_cols if c in protagonist_df.columns]
        print(protagonist_df[available_cols].head(10).to_string(index=False))

    if include_non_protagonist and not non_protagonist_df.empty:
        # Object type breakdown
        obj_types = non_protagonist_df["object_type"].value_counts()
        print("\n    Non-Protagonist Object Types:")
        for otype, count in obj_types.items():
            print(f"      {otype:20s} {count:5d}")


# ============================================================================
# OBJECT EXTRACTION
# ============================================================================

def extract_objects_for_verb(
    verb_token: spacy.tokens.Token,
    char_lookup: Dict[Tuple[int, int], int],
    sentence_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    From a verb token, extract its grammatical objects.

    :param verb_token: The spaCy verb token.
    :param char_lookup: (start_char, end_char) → canonical_id.
    :param sentence_df: DataFrame chunk for this sentence.
    :return: List of object dicts.
    """
    objects: List[Dict[str, Any]] = []

    for child in verb_token.children:
        obj_info = None

        if child.dep_ in ("dobj", "attr"):
            obj_info = _extract_single_object(child, child.dep_, char_lookup, sentence_df)

        elif child.dep_ == "iobj":
            obj_info = _extract_single_object(child, "iobj", char_lookup, sentence_df)

        elif child.dep_ == "prep":
            # Walk through prepositional objects
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    obj_info = _extract_single_object(pobj, "pobj", char_lookup, sentence_df)
                    if obj_info:
                        objects.append(obj_info)

            continue  # pobj already appended

        elif child.dep_ == "dative":
            obj_info = _extract_single_object(child, "iobj", char_lookup, sentence_df)

        if obj_info:
            objects.append(obj_info)

    return objects


def _extract_single_object(
    token: spacy.tokens.Token,
    dep_label: str,
    char_lookup: Dict[Tuple[int, int], int],
    sentence_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Extract info for a single object token, expanding compound nouns."""
    compound_tokens = get_compound_tokens(token)
    text = " ".join(t.text for t in compound_tokens)
    lemma = " ".join(t.lemma_ for t in compound_tokens)

    # Check if this object resolves to a protagonist
    obj_canonical_id = None
    for t in compound_tokens:
        t_row = sentence_df.iloc[t.i]
        t_start = int(t_row["byte_onset"])
        t_end = int(t_row["byte_offset"])
        for (s, e), cid in char_lookup.items():
            if s <= t_start and t_end <= e:
                obj_canonical_id = cid
                break
        if obj_canonical_id is not None:
            break

    # Check possessives for indirect protagonist interactions
    possessor_id = None
    for child in token.children:
        if child.dep_ == "poss":
            c_row = sentence_df.iloc[child.i]
            poss_start = int(c_row["byte_onset"])
            poss_end = int(c_row["byte_offset"])
            for (s, e), cid in char_lookup.items():
                if s <= poss_start and poss_end <= e:
                    possessor_id = cid
                    break

    if obj_canonical_id is not None:
        obj_type = "protagonist"
    else:
        obj_type = "entity"  # will be further classified if needed

    return {
        "object_text": text,
        "object_lemma": lemma,
        "object_type": obj_type,
        "object_canonical_id": obj_canonical_id,
        "possessor_canonical_id": possessor_id,
        "dep_label": dep_label,
    }


# ============================================================================
# NON-PROTAGONIST OBJECT CLASSIFICATION
# ============================================================================

def classify_non_protagonist_object(
    object_text: str,
    gliner_model: Any,
    sentence_text: str,
) -> Dict[str, str]:
    """
    Classify a non-protagonist object using GLiNER and WordNet.

    :return: ``{ entity_label, wordnet_hypernym }``.
    """
    entity_label = "entity"
    wordnet_hypernym = None

    # GLiNER typing
    try:
        labels = ["person", "location", "object", "organization", "concept", "body_part"]
        entities = gliner_model.predict_entities(sentence_text, labels, threshold=0.3)

        for ent in entities:
            if object_text.lower() in ent["text"].lower() or ent["text"].lower() in object_text.lower():
                entity_label = ent["label"]
                break
    except Exception:
        pass

    # WordNet hypernym fallback
    if entity_label == "entity":
        try:
            from nltk.corpus import wordnet as wn
            synsets = wn.synsets(object_text.split()[-1], pos=wn.NOUN)
            if synsets:
                hypernyms = synsets[0].hypernym_paths()
                if hypernyms and len(hypernyms[0]) > 2:
                    wordnet_hypernym = hypernyms[0][-2].name().split(".")[0]
                    # Map to broad categories
                    abstract_roots = {"abstraction", "psychological_feature", "state", "attribute",
                                      "communication", "measure", "relation", "cognition"}
                    if wordnet_hypernym in abstract_roots:
                        entity_label = "abstract"
        except (LookupError, ImportError):
            pass

    return {"entity_label": entity_label, "wordnet_hypernym": wordnet_hypernym}


# ============================================================================
# TRIPLE ASSEMBLY
# ============================================================================

def build_triples(
    verbs_df: pd.DataFrame,
    text: str,
    tokens_df: pd.DataFrame,
    span_index: List[Dict],
    names_df: pd.DataFrame,
    alias_dict: Dict[Any, Any],
    nlp: spacy.language.Language,
    gliner_model: Any,
    include_non_protagonist: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each verb row, extract objects and assemble the final triples.

    :return: ``(protagonist_triples_df, non_protagonist_triples_df)``
    """
    # Build char lookup from span index
    char_lookup = build_char_offset_to_canonical(span_index)

    # Build ID→name and ID→gender lookups
    id_to_name = dict(zip(names_df["id"], names_df["fullname"]))
    id_to_gender = dict(zip(names_df["id"], names_df["gender"]))

    # Collect name variants per canonical ID from alias dict + span index
    id_to_variants: Dict[int, Set[str]] = {}
    for form, info in alias_dict.items():
        # Skip pronouns
        if form.startswith("__pronoun__"):
            continue
        cid = info["canonical_id"]
        if cid not in id_to_variants:
            id_to_variants[cid] = set()
        id_to_variants[cid].add(form)

    print(f'{id_to_name=}')
    print()
    print(f'{id_to_gender=}')
    print()
    print(f'{id_to_variants=}')

    return
    for span in span_index:
        cid = span.get("canonical_id")
        if cid is not None and not span["text"].lower() in PRONOUNS:
            if cid not in id_to_variants:
                id_to_variants[cid] = set()
            id_to_variants[cid].add(span["text"])

    # Process the tokens block by sentence
    protagonist_rows: List[Dict[str, Any]] = []
    non_protagonist_rows: List[Dict[str, Any]] = []

    # Check if verbs_df is empty
    if not verbs_df.empty and "sid" in verbs_df.columns:
        verbs_by_sid = verbs_df.groupby("sid")
    else:
        verbs_by_sid = {}

    # Iterate over tokens_df grouped by sentence_ID and skip all
    # sentences for which a verb has not been previously extracted
    for sid, sentence_df in tokens_df.groupby("sentence_ID"):
        sid = int(sid)
        if sid not in verbs_by_sid.groups:
            continue
        
        # Make spaCy.Doc object for the current sentence
        doc = make_doc_from_sentence(sentence_df, nlp)
        verb_rows_for_sent = verbs_by_sid.get_group(sid)

        for _, verb_row in verb_rows_for_sent.iterrows():
            verb_token_idx = int(verb_row["verb_token_idx"])
            subject_cid = int(verb_row["canonical_id"])

            # Find the verb token in the current doc
            verb_token = None
            for token in doc:
                if token._.global_id == verb_token_idx:
                    verb_token = token
                    break

            if verb_token is None:
                continue

            # Extract objects using DataFrame offsets
            objects = extract_objects_for_verb(verb_token, char_lookup, sentence_df)

            if not objects:
                objects = [{"object_text": "", "object_lemma": "", "object_type": "no_object",
                            "object_canonical_id": None, "possessor_canonical_id": None,
                            "dep_label": ""}]

            for obj in objects:
                # Check if the object is a protagonist
                is_protagonist_obj = obj["object_type"] == "protagonist" and obj["object_canonical_id"] is not None

                # Check indirect interaction via possessive
                if not is_protagonist_obj and obj["possessor_canonical_id"] is not None:
                    is_protagonist_obj = True
                    obj["object_canonical_id"] = obj["possessor_canonical_id"]

                if is_protagonist_obj:
                    right_cid = int(obj["object_canonical_id"])

                    # Skip self-loops
                    if right_cid == subject_cid:
                        continue
                    
                    row = assemble_output_row(
                        subject_cid=subject_cid,
                        verb_text=verb_row["verb_text"],
                        verb_lemma=verb_row["verb_lemma"],
                        verb_index=verb_token_idx,
                        negated=verb_row["negated"],
                        right_cid=right_cid,
                        id_to_name=id_to_name,
                        id_to_gender=id_to_gender,
                        id_to_variants=id_to_variants,
                    )
                    protagonist_rows.append(row)
                    
                elif include_non_protagonist and obj["object_type"] != "no_object":
                    # Classify the non-protagonist object
                    classification = classify_non_protagonist_object(
                        obj["object_text"], gliner_model, doc.text
                    )

                    non_protagonist_rows.append({
                        "canonical_id_left": subject_cid,
                        "name_left": id_to_name.get(subject_cid, ""),
                        "role_left": "agent",
                        "word": verb_row["verb_text"],
                        "lemma": verb_row["verb_lemma"],
                        "index": verb_token_idx,
                        "negated": verb_row["negated"],
                        "gender_left": id_to_gender.get(subject_cid, "u"),
                        "object_text": obj["object_text"],
                        "object_type": classification["entity_label"],
                        "dep_label": obj["dep_label"],
                    })

    protagonist_df = pd.DataFrame(protagonist_rows)
    non_protagonist_df = pd.DataFrame(non_protagonist_rows)

    return protagonist_df, non_protagonist_df


def assemble_output_row(
    subject_cid: int,
    verb_text: str,
    verb_lemma: str,
    verb_index: int,
    negated: bool,
    right_cid: int,
    id_to_name: Dict[int, str],
    id_to_gender: Dict[int, str],
    id_to_variants: Dict[int, Set[str]],
) -> Dict[str, Any]:
    """Assemble a single row in the 14-column target schema."""

    # Limit name variants to only 5 variants
    left_variants = sorted(id_to_variants.get(subject_cid, set()))[:5]
    right_variants = sorted(id_to_variants.get(right_cid, set()))[:5]

    return {
        "canonical_id_left": subject_cid,
        "name_left": id_to_name.get(subject_cid, ""),
        "role_left": "agent",
        "word": verb_text,
        "lemma": verb_lemma,
        "index": verb_index,
        "negated": negated,
        "gender_left": id_to_gender.get(subject_cid, "u"),
        "name_variants_left": str(left_variants),
        "canonical_id_right": right_cid,
        "name_right": id_to_name.get(right_cid, ""),
        "role_right": "patient",
        "gender_right": id_to_gender.get(right_cid, "u"),
        "name_variants_right": str(right_variants),
    }


# ============================================================================
# SAVE
# ============================================================================

def save_stage4(
    triples_df: pd.DataFrame,
    non_protagonist_df: pd.DataFrame | None,
    stage_dir: Path,
    include_non_protagonist: bool = False,
) -> None:
    """Save Stage 4 outputs."""
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Main triples (protagonist-to-protagonist)
    if not triples_df.empty:
        triples_df.to_csv(stage_dir / "triples.csv", sep="\t", index=False)
    else:
        # Write header-only file
        header_cols = [
            "canonical_id_left", "name_left", "role_left", "word", "lemma",
            "index", "negated", "gender_left", "name_variants_left",
            "canonical_id_right", "name_right", "role_right",
            "gender_right", "name_variants_right",
        ]
        pd.DataFrame(columns=header_cols).to_csv(stage_dir / "triples.csv", sep="\t", index=False)

    # Non-protagonist triples (optional)
    if include_non_protagonist and non_protagonist_df is not None and not non_protagonist_df.empty:
        non_protagonist_df.to_csv(stage_dir / "non_protagonist_triples.csv", sep="\t", index=False)


# ============================================================================
# MAIN
# ============================================================================

def main(
    text_path: Path = RAW_TEXT,
    tokens_path: Path = TOKENS,
    coref_dir: Path = COREF_OUT,
    verb_ext_dir: Path = VERB_OUT,
    names_csv: Path = CLEAN_NAMES,
    out_dir: Path = OBJ_OUT,
    include_non_protagonist: bool = False,
    verbose: bool = False,
    nlp=None,
    gliner_model=None,
) -> pd.DataFrame:
    """Run Stage 4: object extraction & final triple assembly."""

    print_headers("STAGE 4 — OBJECT EXTRACTION & TRIPLE ASSEMBLY", "=", prefix="\n")

    # Load inputs
    print_information("Loading inputs...", 1, "\n")
    text = load_text(text_path) # text_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    tokens_df = pd.read_csv(tokens_path, sep="\t", keep_default_na=False)
    names_df = pd.read_csv(names_csv)
    verbs_df = pd.read_csv(verb_ext_dir / "character_verbs.csv", sep="\t")
    print_information(f"Loaded {len(verbs_df)} (character, verb) pairs", prefix="    ")

    # Load span index
    span_index = load_span_index(coref_dir / "span_index.jsonl")
    print_information(f"Loaded {len(span_index)} resolved spans", prefix="    ")

    # Load alias dict
    alias_dict = load_alias_dict(coref_dir / "alias_dict_extended.json")
    print_information(f"Loaded {len(alias_dict)} alias entries", prefix="    ")

    # Load models
    print_information("Loading models...", 2, "\n")
    nlp = load_spacy_model(nlp)

    if include_non_protagonist:
        gliner_model = load_gliner_model(gliner_model)

    # Build triples
    print_information("Building triples...", 3, "\n")
    protagonist_df, non_protagonist_df = build_triples(
        verbs_df, text, tokens_df, span_index, names_df, alias_dict,
        nlp, gliner_model, include_non_protagonist,
    )

    print_information(f"Protagonist-to-protagonist triples: {len(protagonist_df)}", prefix="    ")
    if include_non_protagonist:
        print_information(f"Non-protagonist triples: {len(non_protagonist_df)}", prefix="    ")

    if verbose:
        print_verbose_stats(protagonist_df=protagonist_df, non_protagonist_df=non_protagonist_df, include_non_protagonist=include_non_protagonist, )

    elif not protagonist_df.empty:
        # Even without verbose, show sample
        print("\n    Sample triples:")
        display_cols = ["name_left", "word", "name_right"]
        available_cols = [c for c in display_cols if c in protagonist_df.columns]
        print("    " + protagonist_df[available_cols].head(5).to_string(index=False).replace("\n", "\n    "))

    # Save
    print_information("Saving Stage 4 outputs...", 4, "\n")
    stage_dir = out_dir / "obj_extraction" if out_dir != OBJ_OUT else OBJ_OUT
    save_stage4(protagonist_df, non_protagonist_df, stage_dir, include_non_protagonist)
    print_information(f"Saved to → {stage_dir}", "✓", col="GREEN")

    return protagonist_df


if __name__ == "__main__":
    main(verbose=True)
