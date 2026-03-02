import os
import argparse
import pandas as pd

from rapidfuzz import fuzz, process
from typing import Dict, List, Optional, Set, Tuple
from config import BOOK, BASE_OUT_DIR, TITLES, MAN_CORRECT_IDS, CLEAN_NAMES
from auxiliary import load_and_flatten_characters, normalize_name, clean_non_names, extract_gender, print_headers, print_information


# ============================================================================
# NAME VARIANT GENERATION
# ============================================================================
def build_name_variants(row: pd.Series) -> Set[str]:
    """
    Build all possible name variants for a clean name entry.

    An entity in the book analysed might be referred to by several versions
    of the same name, e.g., by a title and their lastname, a nickname, etc. 
    To ensure, thus, that the various name variants are correctly associated
    with the correct entity, this function pre-computes the set of name variants.

    Generates variants from:
    - `fullname`, `firstname`, `surname`, `middlename`
    - `firstname + surname`, `firstname + middlename`
    - Aliases (from `aka` column)
    - `title + surname`, `title + fullname` (based on gender)
    """
    variants = set()
    titles = TITLES.get(row.get("gender"), [])

    # Core name fields
    fullname = row.get("fullname")
    firstname = row.get("firstname")
    middlename = row.get("middlename")
    surname = row.get("surname")

    # Add individual name components
    for field in [fullname, firstname, middlename, surname]:
        if pd.notna(field) and field:
            variants.add(normalize_name(str(field)))

    # Add combined variants: firstname + surname & firstname + middlename
    if pd.notna(firstname) and pd.notna(surname):
        variants.add(normalize_name(f"{firstname} {surname}"))

    if pd.notna(firstname) and pd.notna(middlename):
        variants.add(normalize_name(f"{firstname} {middlename}"))

    # Add aliases
    aka = row.get("aka")
    if pd.notna(aka) and aka:
        for alias in str(aka).split(";"):
            variants.add(normalize_name(alias.strip()))

    # Add profession
    prof = row.get('profession')
    if pd.notna(prof) and prof:
        for profession in str(prof).split(";"):
            variants.add(normalize_name(profession.strip()))

    # Add title + name variants (only if surname exists)
    if pd.notna(surname) and titles:
        for title in titles:
            variants.add(normalize_name(f"{title} {surname}"))
            if pd.notna(fullname):
                variants.add(normalize_name(f"{title} {fullname}"))

    variants.discard("")
    return variants


def build_variant_index(names_df: pd.DataFrame) -> Tuple[Dict[str, List[int]], List[str], Dict[int, str]]:
    """
    Build lookup structures for efficient matching.

    Returns:
        - variant_to_ids: Dict mapping each variant string -> list of possible IDs
        - all_variants: Flat list of all variants (for rapidfuzz)
        - id_to_gender: Dict mapping ID -> gender ('m', 'f')
    """
    variant_to_ids: Dict[str, List[int]] = {}
    id_to_gender: Dict[int, str] = {}

    for _, row in names_df.iterrows():
        row_id = int(row["id"])
        id_to_gender[row_id] = row["gender"]

        for variant in build_name_variants(row):
            if variant not in variant_to_ids:
                variant_to_ids[variant] = []
            if row_id not in variant_to_ids[variant]:
                variant_to_ids[variant].append(row_id)

    all_variants = list(variant_to_ids.keys())
    
    # Also build a set of all individual name tokens for phrase extraction
    all_name_tokens = set()
    for variant in all_variants:
        all_name_tokens.update(variant.split())
    
    return variant_to_ids, all_variants, id_to_gender, all_name_tokens


# ============================================================================
# MATCHING LOGIC
# ============================================================================
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

    :param name: The name to match (will be normalized)
    :param gender: Gender hint ('m', 'f', 'u') for disambiguation
    :param variant_to_ids: Mapping from variant string to list of IDs
    :param all_variants: List of all variant strings
    :param id_to_gender: Mapping from ID to gender
    :param names_df: DataFrame with clean names (for fullname lookup)
    :param all_name_tokens: Set of known name tokens for phrase extraction
    :param threshold: Minimum score (0-100) to accept a match
    :return: (matched_id, matched_fullname, score) or (None, original_name, 0.0)
    """
    # Preprocess current name
    original_name = name  # Keep for returning if unmatched
    normalized = normalize_name(name)
    cleaned = clean_non_names(normalized, all_name_tokens)

    if not cleaned:
        return None, original_name, 0.0, None

    # Helper function to try matching a normalized string
    def try_fuzzy_match(query: str) -> Tuple[Optional[int], Optional[str], float, Optional[str]]:
        """
        Uses the Levenshtein distance compare the similarity between two tokenized
        lists of names.
        
        :param query: The name to match amongst the name variants
        :type query: str
        :param enforce_gender: Whether or not to enforce that the genders need to match
        :type enforce_gender: bool
        :return: A tuple with the character's ID, fullname, and the socre of the best match
        :rtype: Tuple[int | None, str | None, float]
        """
        matches = process.extract(
            query, all_variants, scorer=fuzz.token_sort_ratio, limit=10
        )

        # print(query, "\t", matches)
        for variant, score, _ in matches:
            # Stop considering matches if the score 
            # goes below the set threshold
            if score < threshold:
                break

            candidate_ids = variant_to_ids[variant]

            for cid in candidate_ids:
                db_gender = id_to_gender.get(cid, "u")

                # Don't process names with unknown gender
                if gender == "u":
                    continue
                
                # Correct the gender of an entry, if there is a
                # mismatch between that name's gender value in the
                # DataFrame with clean names
                corrected_gender = None
                if gender in TITLES.keys() and gender != db_gender:
                    # Penalise having to adjust the gender
                    penalty = 1.0 if gender == db_gender else 0.8
                    if (score * penalty) >= threshold:
                        corrected_gender = db_gender
                    else:
                        continue

                fullname = names_df.loc[names_df["id"] == cid, "fullname"].iloc[0]
                    
                return cid, fullname, score, corrected_gender

        return None, None, 0.0, None

    # Try matching with gender enforcement first
    for enforce_gender in [True]:
        # Try direct match with cleaned name
        cid, fullname, score, corrected_gender = try_fuzzy_match(cleaned)

        if cid is not None:
            # Apply penalty if we had to ignore gender
            penalty = 1.0 if enforce_gender else 0.8
            return cid, fullname, score * penalty, corrected_gender

        # If still no match, try matching individual tokens
        # This handles single first names like "Hugo" -> "Hugo Hamilton"
        tokens = cleaned.split()
        for token in tokens:
            if token in all_name_tokens and len(token) > 2:  # Skip very short tokens
                cid, fullname, score, corrected_gender = try_fuzzy_match(token)
                if cid is not None:
                    # Reduce score slightly since we're matching only a partial name
                    penalty = 0.9 if enforce_gender else 0.7  # Additional penalty for gender mismatch
                    return cid, fullname, score * penalty, corrected_gender 

    return None, original_name, 0.0, None


def merge_final_output(characters: pd.DataFrame, canonical_summary: pd.DataFrame, output_dir: str):
    """
    Merges canonical_mappings.csv and matched_characters.csv into a single file of the following format.

    canonical_id, name, role, word, index, original_ids, gender, name_variants, avg_score
    Description:
        canonical_id: int   --> The canonical ID of the character in that entry
        name: str           --> The canonicalised name of the character in that entry
        role: str           --> Whether or not the character is an 'agent' or 'patient' in that entry
        word: str           --> The word of action of that entry
        index: int          --> The index/token ID of the word of that entry
        original_ids: List  --> A list of the original character IDs that have been merged into the canonical ID
        gender: str         --> Whether the character is a (m)ale or (f)emale
        name_variants: List --> A list of the original names associated with a character
        avg_score: float    --> The average confidence score for the match of names to the canonical name of that entry
    """
    # TODO: POTENTIALLY MOVE TO auxiliary.py TO BE REUSED AFTER SVO EXTRACTION
    print_information("Merging content for final output...", 7, "\n")

    # Group characters to get variants per canonical_id
    # We want unique original names associated with each canonical_id
    variants = (
        characters
        .groupby("canonical_id")["original_name"]
        .unique()
        .apply(list)
        .rename("name_variants")
        )

    # Merge variants into summary
    canonical_data = canonical_summary.merge(variants, on="canonical_id", how="left")

    # Merge canonical data into characters
    # characters has [canonical_id, names, role, word, index, gender, original_name, match_score, etc.]
    # canonical_data has [canonical_id, fullname, original_ids, avg_name_match_score, name_variants]
    
    merged = characters.merge(canonical_data, on="canonical_id", how="left")

    # Construct the final DataFrame with selected columns
    output_df = pd.DataFrame({
        'canonical_id': merged['canonical_id'],
        'name': merged['names'],
        'role': merged['role'],
        'word': merged['word'],
        'index': merged['index'],
        'original_ids': merged['original_ids'],
        'gender': merged['gender_hint'],
        'name_variants': merged['name_variants'],
    })

    # Drop duplicates to ensure one entry per event (deduplicating name variants)
    output_df = output_df.drop_duplicates(subset=['canonical_id', 'role', 'word', 'index'])
    
    # Remove entries whose canonical ID achieved a match score of 0 i.e.,
    # they weren't matched to one of the valid names available
    zero_match_score_ids = canonical_summary[canonical_summary['avg_name_match_score'] == 0]['canonical_id'].unique()
    output_df = output_df[~output_df['canonical_id'].isin(zero_match_score_ids)]

    # Save to tab-delimited CSV
    output_filename = "merged_characters.characters"
    output_path = os.path.join(output_dir, output_filename)
    output_df.to_csv(output_path, sep='\t', index=False)
    
    print_information(f"Final merged output saved to: {output_path}", "✓", col="GREEN")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main(input_file: str, output_dir: str, verbose: bool = False):
    print_headers("RUNNING NAME MATCHING PIPELINE", "=", "\n")

    # Load clean names
    print_information("Loading clean names database...", 1, "\n")
    names_df = pd.read_csv(CLEAN_NAMES)
    print_information(f"Loaded {len(names_df)} canonical characters", prefix="    ")

    # Build index of name variant
    print_information("Building name variant index...", 2, "\n")
    variant_to_ids, all_variants, id_to_gender, all_name_tokens = build_variant_index(names_df)
    print_information(f"Generated {len(all_variants)} unique variants", prefix="    ")


    # Load extracted characters from BookNLP output
    print_information(f"Loading extracted characters from: {input_file}", 3, "\n")
    characters = load_and_flatten_characters(input_file, verbose=verbose)
    print_information(f"Loaded {len(characters)} (exploded) character mentions", prefix="    ")

    # Extract the gender from BookNLP-extracted pronouns
    print_information("Extracting gender information from pronouns...", 4, "\n")
    characters["gender_hint"] = characters["gender"].apply(extract_gender)

    # Match each name to the clean names
    print_information("Matching names to canonical database...", 5, "\n")

    matched_ids = []
    matched_fullnames = []
    match_scores = []
    corrected_genders = []

    for _, row in characters.iterrows():
        name = row["names"]
        gender_hint = row["gender_hint"]

        matched_id, fullname, score, corrected_gender = match_name(
            name, gender_hint, variant_to_ids, all_variants, id_to_gender, names_df, all_name_tokens
        )

        matched_ids.append(matched_id)
        matched_fullnames.append(fullname)
        match_scores.append(score)
        corrected_genders.append(corrected_gender)

    characters["matched_id"] = matched_ids
    characters["matched_fullname"] = matched_fullnames
    characters["match_score"] = match_scores
    characters["corrected_gender"] = corrected_genders

    # Manually correct matching of the following IDs
    indeces = characters[characters['character_id'].isin(MAN_CORRECT_IDS)].index
    characters.loc[indeces, ['matched_id', 'matched_fullname', 'match_score']] = None, 'God', 0.0

    # Update the 'names' column with the matched fullname (keeps original if name is unmatched)
    characters["original_name"] = characters["names"]
    characters["names"] = characters.apply(
        lambda r: r["matched_fullname"] if r["matched_id"] is not None else r["original_name"],
        axis=1
    )
    print_information("Matched all names", prefix="    ")

    # Canonicalise the character IDs by finding the smallest
    # original character_id for each matched_fullname
    print_information("Consolidating character IDs...", 6, "\n")

    # Get all entries that where matched to a name variant &
    # generate their canonical ID
    canonical_ids = {}
    matched_mask = characters["matched_id"].notna()
    for fullname in characters.loc[matched_mask, "matched_fullname"].unique():
        # Get all original character_ids for this matched name
        char_ids = characters.loc[
            (matched_mask) & (characters["matched_fullname"] == fullname), "character_id"
        ].unique()
        canonical_ids[fullname] = min(char_ids)

    # Write back canonical IDs to DataFrame
    characters["canonical_id"] = characters.apply(
        lambda r: canonical_ids.get(r["matched_fullname"], r["character_id"]),
        axis=1
    )

    # Print summary and save results
    if verbose:
        print_headers("MATCHING SUMMARY", "-", prefix="\n")

        total_mentions = len(characters)
        match_mask = characters["matched_id"].notna()
        matched = match_mask.sum()
        unmatched = (~match_mask).sum()
        unique_matched = characters.loc[characters["matched_id"].notna(), "matched_fullname"].nunique()
        unique_canonical = characters["canonical_id"].nunique()
        num_corrected_genders = characters["corrected_gender"].notna().sum()

        print(f"    Total mentions:         {total_mentions}")
        print(f"    Matched entries:        {matched} ({matched / total_mentions * 100:.1f}%)")
        print(f"    Unmatched entries:      {unmatched} ({unmatched / total_mentions * 100:.1f}%)")
        print(f"    Unique matched names:   {unique_matched}")
        print(f"    Unique canonical IDs:   {unique_canonical} ({unique_canonical / characters['character_id'].nunique() * 100:.1f}%)")
        print(f"    Genders corrected:      {num_corrected_genders} ({num_corrected_genders / total_mentions * 100:.1f}%)")
        print(f"    Avg match score:        {characters['match_score'].mean():.1f}")

        # Show sample of unmatched name variants
        if unmatched > 0:
            print_information("Unmatched name variants:", prefix="\n    ")
            unmatched_names = (
                characters.loc[
                    characters["match_score"].astype(int) == 0, 
                    "original_name"
                    ].value_counts()
                    )
            
            for name, count in unmatched_names.items():
                print_information(f"{name} ({count} mentions)", prefix="      - ")

    # Save results
    output_path = f"{output_dir}/matched_characters.csv"
    characters.to_csv(output_path, index=False)
    print_information(f"Results saved to: {output_path}", symb="✓", prefix="\n")

    # Also save a summary of canonical mappings
    canonical_summary = characters.groupby("canonical_id").agg({
        "matched_fullname": "first",
        "character_id": lambda x: sorted(set(x)),
        "match_score": "mean"
    }).reset_index()
    canonical_summary.columns = ["canonical_id", "fullname", "original_ids", "avg_name_match_score"]

    # 
    summary_path = f"{output_dir}/canonical_mappings.csv"

    if verbose:
        print_headers("OVERVIEW", "-", prefix="\n")
        print(canonical_summary.to_string(index=False))

    canonical_summary.to_csv(summary_path, index=False)
    print_information(f"Canonical mappings saved to: {summary_path}", symb="✓")

    merge_final_output(characters, canonical_summary, output_dir)
    return characters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match extracted character names to a canonical database and consolidate IDs."
    )
    parser.add_argument(
        "-i", "--input",
        default=BOOK,
        help="Path to the .book JSON file from BookNLP"
    )
    parser.add_argument(
        "-o", "--output",
        default=BASE_OUT_DIR,
        help="Output directory for results"
    )

    parser.add_argument(
        '-v', '--verbose', 
        action="store_true",
        help="A flag to trigger verbose output"
        )

    args = parser.parse_args()
    main(args.input, args.output, args.verbose)
