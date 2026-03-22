"""
Configuration for the manual extraction pipeline.
Standalone — does NOT import from src.config.
"""
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================
# INPUT PATHS
# ============================
RAW_TEXT    = _PROJECT_ROOT / "data" / "book" / "attwn_2_chpts.md"
CLEAN_NAMES = _PROJECT_ROOT / "data" / "names_owen_split.csv"

# ============================
# OUTPUT
# ============================
OUT_DIR = _PROJECT_ROOT / "out_manual"
CHAR_RES_OUT = OUT_DIR / "char_resolution"
COREF_OUT = OUT_DIR / "coreference"
VERB_OUT = OUT_DIR / "verb_extraction"
OBJ_OUT = OUT_DIR / "object_extraction"
TOKENS = OUT_DIR / RAW_TEXT.name.replace(".md", ".tokens")

# ============================
# CONSTANTS
# ============================
TITLES = {"m": ["mr", "mister"], "f": ["mrs", "ms", "miss"]}
ARTICLES = ["the", "a", "an"]
PRONOUNS = {"he", "him", "his", "she", "her", "hers", "they", "them",
            "their", "it", "its", "i", "me", "my", "we", "us", "our",
            "himself", "herself", "myself", "ourselves", "themselves"}

NON_NAME_WORDS = set(ARTICLES) | {
    # Determiners & articles
    "this", "that", "these", "those",
    # Adjectives commonly used with names
    "old", "young", "poor", "dear", "good", "little", "great", "late",
    "fellow", "chap", "man", "woman", "lady", "gentleman", "boy", "girl",
    # Adverbs/modifiers
    "suddenly", "nevertheless", "only", "just", "even", "perhaps", "certainly",
    # Possessives
    "my", "your", "his", "her", "their", "our",
    # Other common non-name words
    "one", "someone", "anyone", "nobody", "everybody",
    "died", "who", "dearest", "bloody",
}

# ============================
# MODEL SETTINGS
# ============================
# SPACY_MODEL = "data/models/en_core_web_trf"
SPACY_MODEL = "data/models/en_core_web_sm"

# ============================
# COREF WINDOW SETTINGS
# ============================
COREF_WINDOW_TOKENS  = 512
COREF_OVERLAP_TOKENS = 128

# ============================
# FUZZY MATCH THRESHOLDS
# ============================
FUZZY_AUTO_ACCEPT = 90.0   # auto-accept above this
FUZZY_REVIEW_LOW  = 70.0   # flag for review between LOW..AUTO_ACCEPT
