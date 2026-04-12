from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent #.parent

print(f'PROJECT_ROOT: {_PROJECT_ROOT}')

# ============================
# INPUT PATHS
# ============================
RAW_TEXT    = _PROJECT_ROOT / "data" / "book" / "attwn.md"
CLEAN_NAMES = _PROJECT_ROOT / "data" / "names_owen_split.csv"

# ============================
# OUTPUT
# ============================
OUT_DIR = _PROJECT_ROOT / "out"
DATA_DIR = _PROJECT_ROOT / "data"
CHAR_RES_OUT = OUT_DIR / "char_resolution"
COREF_OUT = OUT_DIR / "coreference"
VERB_OUT = OUT_DIR / "verb_extraction"
OBJ_OUT = OUT_DIR / "object_extraction"
TOKENS = OUT_DIR / RAW_TEXT.name.replace(".md", ".tokens")
EGO_OUT = OUT_DIR / "ego_networks"

# ============================
# CONSTANTS
# ============================

# A dictionary of sex abbreviations to titles
TITLES = {"m": ["mr", "mister"], "f": ["mrs", "ms", "miss"]}

# A list of English language articles. Used for removal of them
ARTICLES = ["the", "a", "an"]


PRONOUNS = {"he", "him", "his", "she", "her", "hers", "they", "them",
            "their", "it", "its", "i", "me", "my", "we", "us", "our",
            "himself", "herself", "myself", "ourselves", "themselves"}

# Common words that appear in phrases but are not proper names
# These will be stripped when trying to extract a proper name from a phrase
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

DEATH_INTERVALS = [
    ("#1 Marston",     1,    1292, 10), # 1, 1484
    ("#2 E.Rogers",    1293, 1609, 9), # 1485, 1841
    ("#3 Macarthur",   1610, 2486, 8), # 1842, 2904
    ("#4 T.Rogers",    2487, 3057, 7), # 2905, 3583 
    ("#5 Brent",       3058, 3344, 6), # 3584, 3933
    ("#6 Wargrave",    3345, 3749, 5), # 3934, 4416
    ("#7 Blore",       3750, 4314, 4), # 4417, 5096
    ("#8 Armstrong",   4315, 4393, 3), # 5097, 5195
    ("#9 Lombard",     4394, 4455, 2), # 5196, 5279
    ("#10 Claythorne", 4456, 4544, 1), # 5280, 5386
]

INTERVALS = [
    ("#1 Marston",     1, 1484, 10),
    ("#2 E.Rogers",    1485, 1841, 9),
    ("#3 Macarthur",   1842, 2904, 8), 
    ("#4 T.Rogers",    2905, 3583, 7), 
    ("#5 Brent",       3584, 3933, 6),
    ("#6 Wargrave",    3934, 4416, 5),
    ("#7 Blore",       4417, 5096, 4),
    ("#8 Armstrong",   5097, 5195, 3),
    ("#9 Lombard",     5196, 5279, 2),
    ("#10 Claythorne", 5280, 5386, 1)
]
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

# ============================
# Visualisation Settings
# ============================
EGO_COLOR = "#e63946"
ALTER_COLOR = "#2a9d8f"
