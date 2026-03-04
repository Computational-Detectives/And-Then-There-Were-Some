from pathlib import Path

# ============================
# COMMON FILE PATHS
# ============================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard output location
BASE_DATA_DIR = str(_PROJECT_ROOT / "data")
BASE_OUT_DIR = str(_PROJECT_ROOT / "out")
TRIPLE_OUT = BASE_OUT_DIR + "/triples"
COOC_OUT = BASE_OUT_DIR + "/cooccurrence"
NETWORK_OUT = BASE_OUT_DIR + "/network"
EGO_OUT = BASE_OUT_DIR + "/ego_networks"

# Path to BookNLP files
ATTWN = BASE_DATA_DIR + "/book/attwn.txt"
BOOK = BASE_OUT_DIR + "/preproc_attwn.book"
ENTITY = BASE_OUT_DIR + "/preproc_attwn.entities"
TOKENS = BASE_OUT_DIR + "/preproc_attwn.tokens"

# Path to canonical names database
CLEAN_NAMES = BASE_DATA_DIR + "/names_owen_split.csv"


# ============================
# CONSTANTS
# ============================

# A dictionary of sex abbreviations to titles
TITLES = {"m": ["mr", "mister"], "f": ["mrs", "ms", "miss"]}

# A list of English language articles. Used for removal of them
ARTICLES = ["the", "a", "an"]

# A list of IDs of entries w/ 'common' values to retain & match against names
KEEP_IDS_COMMON = [287, 338, 459, 737, 861, 913, 1084, 1635, 1689, 1730, 1741, 2045, 2051, 2869, 2873]

# A list of IDs for which to manually add gender information
ADD_GENDER_IDS = [737, 861, 1084]

# A list of IDs for which to manually correct/reversing the name matching
MAN_CORRECT_IDS = [83, 135]

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
    # Other common non-name words in the corpus
    "one", "someone", "anyone", "nobody", "everybody",
    "died", "who", "dearest", "bloody"
}