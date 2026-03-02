import os
import re
import ast
import csv
import json
import spacy
import pandas as pd

from pathlib import Path
from colorama import Fore, Style, init
from typing import List, Set, Optional
from config import ARTICLES, NON_NAME_WORDS, KEEP_IDS_COMMON, ADD_GENDER_IDS, BASE_OUT_DIR

init()

# ============================
# Preprocessing
# ============================
def preprocess(input_file: str, output_file: str):
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Read raw text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Create first match pattern
    headers_pttrn = re.compile(
        # r'\bChapter\s+\d+\b|'                               # Match Chapter headers
        # r'\bEpilogue\b|'                                    # Match Epilogue header
        # r'\bA MANUSCRIPT DOCUMENT SENT TO SCOTLAND YARD\b|' # Match Manuscript header
        # r'\b[IVXLCDM]+\b\s+|'                               # Match Roman numerals
        r"\bIll\b\s+"  # Match ill-formatted Roman numeral III
    )

    cleaned_text = headers_pttrn.sub("", text)

    # Normalize line endings
    text = cleaned_text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse mid-sentence line breaks outside quotes
    # Replace single newlines inside paragraphs with space, keep double newlines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    # Segment into individual sentences
    doc = nlp(text)

    # Write collapsed sentences to file
    if not os.path.exists(Path(output_file).parent):
        os.mkdir(Path(output_file).parent)

    with open(output_file, "w", encoding="utf-8") as f:
        for sent in doc.sents:
            # Collapse everything inside the sentence
            clean_sent = re.sub(r"\s+", " ", sent.text).strip()
            f.write(clean_sent + "\n")


def load_booknlp_file(path: str):
    return pd.read_csv(
        path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        engine="python",
        keep_default_na=False
    )

    
def load_and_flatten_characters(input_file: str, verbose: bool = False) -> pd.DataFrame:
    """
    This method loads the data on the extracted characters from the JSON
    output of the `BookNLP` pipeline and recreates the structure of the
    most necessary information for further processing as a `pd.DataFrame`.

    It extracts values both from the `proper` and `common` JSON arrays

    :param input_file: The path to the JSON file
    :type input_file: str
    :param verbose: Whether to print additional information
    :type verbose: bool
    :return: The flattened DataFrame containing the most necessary character information
    :rtype: DataFrame
    """

    # Load `.book`-output file for further processing
    with open(input_file, "r") as f:
        data = json.load(f)

    # Extract high-level Character-object from JSON file
    characters_lst = data["characters"]
    rows = []

    # For each character...
    for char in characters_lst:
        # ...extract their character_ID
        cid = char["id"]
        # ...for each role they played
        for role in ["agent", "patient"]:  # , "mod", "poss"]:
            # ...extract the action & index associated with it
            for tok in char.get(role, []):
                rows.append(
                    {
                        "character_id": cid,
                        "names": [elem["n"] for elem in char["mentions"]["proper"]],
                        "common": [elem["n"] for elem in char["mentions"]["common"]],
                        "role": role,
                        "word": tok["w"],
                        "index": tok["i"],
                        "gender": "" if char["g"] is None else char["g"]["argmax"],
                    }
                )

    characters = pd.DataFrame(rows)

    if verbose:
        # Collect statistics
        total_entries = characters.shape[0]
        num_names = characters[~characters['names'].isin([[], '[]'])].shape[0]
        num_common = characters[~characters['common'].isin([[], '[]'])].shape[0]

    # Apply filters
    # Remove all entries w/o 'names' or 'common'
    characters = (
        characters[
            (~characters['names'].isin([[], '[]'])) |
            (~characters['common'].isin([[], '[]']))
            ]
        )
    
    # Remove all entries where 'common' is not of interest
    characters = (
        characters[
            (~characters['names'].isin([[], '[]'])) |
            ((~characters['common'].isin([[], '[]'])) &
            (characters['character_id'].isin(KEEP_IDS_COMMON)))
            ]
        )
    
    # Add gender for characters w/ ID 737, 861, 1084
    characters.loc[characters['character_id'].isin(ADD_GENDER_IDS), 'gender'] = 'he/him/his'

    # Merge retained 'common' entries into names
    characters['names'] = characters['names'] + characters['common']
    characters = characters.drop(columns=['common'])

    # ========== STATISTICS ==========
    if verbose:
        print()
        print_headers("Summary of .book Extraction", "-")

        print(f"    Total entries:                  {total_entries}")
        print(f"    # entries w/ names:             {num_names} ({num_names / total_entries * 100:.1f}%)")
        print(f"    # entries w/ common:            {num_common} ({num_common / total_entries * 100:.1f}%)")       
        print(f"    # Total entries after removed:  {characters.shape[0]} ({characters.shape[0] / total_entries * 100:.1f}%)\n")

    # Main DataFrame: Explode names list onto individual row for better post-processing
    characters = characters.explode("names")
    return characters


def safe_to_list(x):
    """Convert value to list, handling various input types."""
    if isinstance(x, list):
        return x
    if isinstance(x, (set, tuple)):
        return list(x)
    if isinstance(x, str):
        try:
            result = ast.literal_eval(x)
            return list(result) if isinstance(result, (list, set, tuple)) else []
        except (ValueError, SyntaxError):
            return []
    return []


# ============================
# NORMALIZATION UTILITIES
# ============================
def normalize_name(name: str) -> str:
    """
    Normalize a name string for consistent matching.

    - Converts to lowercase
    - Removes punctuation (periods, commas, hyphens)
    - Collapses multiple spaces
    - Removes leading articles (the, a, an)
    """
    if pd.isna(name) or not name:
        return ""

    name = name.lower()
    name = re.sub(r"[.,\"-]", "", name)
    name = " ".join(name.split())

    for article in ARTICLES:
        if name.startswith(article + " "):
            name = name[len(article):].strip()

    return name


def clean_non_names(normalized_phrase: str, known_names: Set[str]) -> str:
    """
    Extract proper name tokens from a phrase by removing non-name words.
    
    For example:
    - "That fellow Lombard" -> "Lombard"
    - "Poor Dr. Armstrong" -> "Dr Armstrong"
    - "Nevertheless Blore" -> "Blore"
    
    :param phrase: The phrase to extract names from
    :param known_names: Set of known proper name tokens (lowercase)
    :return: Cleaned string with only potential name parts
    """
    if not normalized_phrase:
        return ""
    
    # Normalize first
    tokens = normalized_phrase.split()
    
    # Keep tokens that are:
    # - known i.e., name variant, OR
    # - start with a capital letter in the original (proper noun indicator), OR  
    # - are titles (mr, mrs, dr, etc.)
    title_words = {"mr", "mrs", "ms", "miss", "mister", "dr", "captain", "general", 
                   "inspector", "sir", "lady", "justice"}
    
    kept_tokens = []
    for token in tokens:
        # Keep if it's a known name or title
        if token in known_names or token in title_words:
            kept_tokens.append(token)
        # Remove if it's a known non-name word
        elif token in NON_NAME_WORDS:
            continue
        # Keep if it's not a common English word
        elif len(token) > 2:
            kept_tokens.append(token)
    
    return " ".join(kept_tokens)


def extract_gender(pronoun_str: str) -> str:
    """
    Extract gender category from pronoun string.

    :param pronoun_str: String like 'he/him/his' or 'she/her/hers'
    :return: 'm' for male, 'f' for female, or 'u' for unknown
    """
    if not pronoun_str or pronoun_str == "they/them/their":
        return "u"

    pronoun_str = pronoun_str.lower()

    # Check female FIRST since "she" contains "he" as a substring   
    if any(p in pronoun_str for p in ["she", "her"]):
        return "f"
    elif any(p in pronoun_str for p in ["he", "him", "his"]):
        return "m"
    return "u"
    

# ========================
# Print Helper
# ========================
def print_headers(msg: str, symb: str, prefix: str = ""):
    print(f'{prefix}{symb * 80}\n{msg}\n{symb * 80}')


def print_information(msg: str, symb: Optional[str|int] = None, prefix: str = "", col: str = "BLUE"):
    if symb:
        colour = getattr(Fore, col)
        print(f'{prefix}[{colour}{symb}{Style.RESET_ALL}] {msg}')
    else:
        print(f'{prefix}{msg}')


# ========================
# ArgParser
# ========================
def get_chapter_token_range() -> str:
    """
    An auxiliary function to get the token ranges for each chapter of the analysed book.

    The token range can be used to filter the tokenized text for a specific chapter and focus
    further analysis solely on that range.

    :return: A tabular string representation of the token ranges for each chapter.
    :rtype: str
    """
    tokens = pd.read_csv(f"{BASE_OUT_DIR}/preproc_attwn.tokens", sep="\t")

    # Filter rows where word matches our chapter markers
    chapter_markers = tokens[
        (tokens["word"] == "Chapter")
        | (tokens["word"] == "Epilogue")
        | (tokens["word"] == "MANUSCRIPT")
    ].copy()

    # Extract token IDs & shift last chapter token back by one token
    token_ids = chapter_markers["token_ID_within_document"].tolist()
    token_ids[-1] = token_ids[-1] - 1

    # Build chapter ranges
    chapters = []
    for i, token_id in enumerate(token_ids):
        if i < 16:
            start_token = token_id + 3
        elif i == 16:
            start_token = token_id + 1
        elif i == 17:
            start_token = token_id + 18
        else:
            start_token = token_id

        # End token is the token before the next chapter starts, or last token in document
        if i < len(token_ids) - 1:
            end_token = token_ids[i + 1] - 1
        else:
            end_token = tokens["token_ID_within_document"].max()

        chapters.append(
            {"Chapter": i + 1, "Start Token": start_token, "End Token": end_token}
        )

    # Create DataFrame for easy formatting
    df = pd.DataFrame(chapters)

    # Format as a table string
    # Calculate column widths
    col1_width = max(len("Chapter"), max(len(str(x)) for x in df["Chapter"]))
    col2_width = max(len("Start Token"), max(len(str(x)) for x in df["Start Token"]))
    col3_width = max(len("End Token"), max(len(str(x)) for x in df["End Token"]))

    # Build table
    header = f"{'Chapter':<{col1_width}} | {'Start Token':<{col2_width}} | {'End Token':<{col3_width}}"
    separator = f"{'-' * col1_width}-+-{'-' * col2_width}-+-{'-' * col3_width}"

    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{str(row['Chapter']):<{col1_width}} | {str(row['Start Token']):<{col2_width}} | {str(row['End Token']):<{col3_width}}"
        )

    table = "\n".join([header, separator] + rows)

    return table


def int_range(args: List[str]) -> List[int]:
    """
    An auxiliary function for custom type specification
    for the argument parser.

    :param args: The argument passed through the CLI to be processed
    :type: List[str]
    :return: A list of integers to define a token range
    :rtype: List[int]
    """
    return [int(a) for a in args.split(",")]
