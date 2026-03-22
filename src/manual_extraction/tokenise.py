"""
Stage 0 — Tokenisation

Produces a .tokens file with a 12-column tab-separated schema
matching BookNLP's output (minus `event`), using spaCy.
"""
from __future__ import annotations

from pathlib import Path
from tqdm import tqdm

from .config import OUT_DIR, RAW_TEXT, SPACY_MODEL, TOKENS
from .utils import print_headers, print_information, load_spacy_model, load_text


def save_tokens(df: pd.DataFrame, text_path: Path, outdir: Path) -> Path:
    """
    Write the tokenised DataFrame to a tab-separated .tokens file.

    :param df: The token DataFrame.
    :return: Path to the written file.
    """
    out_path = outdir / text_path.name.replace(".md", ".tokens")
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def print_verbose_stats(tokens: pd.DataFrame, n_tokens: int):
    """
    Print verbose statistics about the tokenisation.

    :param tokens: The token DataFrame.
    :param n_tokens: The total number of tokens.
    """
    print_headers("TOKENISATION STATISTICS", "-", prefix="\n")

    # POS tag distribution
    pos_counts = tokens["POS_tag"].value_counts()
    print("    POS Tag Distribution (top 10):")
    for pos, count in pos_counts.head(10).items():
        pct = count / n_tokens * 100
        print(f"      {pos:12s} {count:6d}  ({pct:5.1f}%)")

    # Dependency relation distribution
    print("\n    Dependency Relation Distribution (top 10):")
    dep_counts = tokens["dependency_relation"].value_counts()
    for dep, count in dep_counts.head(10).items():
        pct = count / n_tokens * 100
        print(f"      {dep:12s} {count:6d}  ({pct:5.1f}%)")

    # Sentence length distribution
    sent_lengths = tokens.groupby("sentence_ID").size()
    print(f"\n    Sentence length — min: {sent_lengths.min()}, "
            f"max: {sent_lengths.max()}, "
            f"median: {sent_lengths.median():.0f}, "
            f"mean: {sent_lengths.mean():.1f}")


def tokenise_text(text: str, nlp: spacy.language.Language) -> pd.DataFrame:
    """
    Process the full text with spaCy and emit a 12-column DataFrame
    matching the BookNLP ``.tokens`` format (without ``event``).

    Columns:
        paragraph_ID, sentence_ID, token_ID_within_sentence,
        token_ID_within_document, word, lemma, byte_onset, byte_offset,
        POS_tag, fine_POS_tag, dependency_relation, syntactic_head_ID

    :param text: The full raw text.
    :param nlp: A loaded spaCy Language model.
    :return: DataFrame with one row per token.
    """
    # Detect paragraph boundaries by splitting on double-newlines
    paragraphs = text.split("\n\n")

    rows: list[dict] = []
    global_token_id = 0
    global_sent_id = 0
    char_offset = 0  # running offset into the original text

    import spacy
    for para_id, para_text in tqdm(enumerate(paragraphs), total=len(paragraphs)):
        if not para_text.strip():
            # Account for the double-newline separator
            char_offset += len(para_text) + 2
            continue

        doc = nlp(para_text)

        # Map spaCy token index within this paragraph to global output token ID
        spacy_i_to_global: dict[int, int] = {}
        for token in doc:
            if not token.is_space:
                spacy_i_to_global[token.i] = global_token_id
                global_token_id += 1

        # Track sentence boundaries within this paragraph
        for sent in doc.sents:
            token_id_within_sent = 0
            has_tokens = False
            for token in sent:
                if token.is_space:
                    continue

                has_tokens = True
                # Character offsets relative to the full original text
                byte_onset = char_offset + token.idx
                byte_offset = byte_onset + len(token.text)

                # syntactic_head_ID in global document-level token ID
                head_spacy_i = token.head.i
                # If head is a space token (rare but possible), self-reference as fallback
                head_global_id = spacy_i_to_global.get(head_spacy_i, spacy_i_to_global[token.i])

                rows.append({
                    "paragraph_ID": para_id,
                    "sentence_ID": global_sent_id,
                    "token_ID_within_sentence": token_id_within_sent,
                    "token_ID_within_document": spacy_i_to_global[token.i],
                    "word": token.text,
                    "lemma": token.lemma_,
                    "byte_onset": byte_onset,
                    "byte_offset": byte_offset,
                    "POS_tag": token.pos_,
                    "fine_POS_tag": token.tag_,
                    "dependency_relation": token.dep_,
                    "syntactic_head_ID": head_global_id,
                })

                token_id_within_sent += 1

            if has_tokens:
                global_sent_id += 1

        # Account for paragraph text + double-newline separator
        char_offset += len(para_text) + 2

    import pandas as pd
    return pd.DataFrame(rows)
    

def main(text_path: Path = RAW_TEXT, out_dir: Path = OUT_DIR, verbose: bool = False, nlp=None) -> Path:
    """Tokenise the raw text and save .tokens file."""
    print_headers("STAGE 0 — TOKENISATION", "=", prefix="\n")

    # Load text
    print_information("Loading raw text...", 1, "\n")
    text = load_text(text_path) # text_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    print_information(f"Loaded {len(text)} characters from {text_path.name}", prefix="    ")

    # Load spaCy model
    print_information(f"Loading spaCy model '{SPACY_MODEL}'...", 2, "\n")
    nlp = load_spacy_model(nlp)

    # Tokenise
    import pandas as pd
    print_information("Tokenising text per paragraph...", 3, "\n")
    tokens = tokenise_text(text, nlp)

    # Prepare statistics
    n_tokens = len(tokens)
    n_sents = tokens["sentence_ID"].nunique()
    n_paras = tokens["paragraph_ID"].nunique()
    avg_tok_per_sent = n_tokens / n_sents if n_sents else 0

    print_information(f"Produced {n_tokens} tokens across {n_sents} sentences in {n_paras} paragraphs", prefix="    ")
    print_information(f"Avg tokens per sentence: {avg_tok_per_sent:.1f}", prefix="    ")

    # Print verbose statistics
    if verbose:
        print_verbose_stats(tokens, n_tokens)

    # Save tokens to '.tokens'
    print_information("Saving .tokens file...", 4, "\n")
    out_path = save_tokens(tokens, text_path, out_dir)
    print_information(f"Saved to → {out_path}", "✓", col="GREEN")

    return out_path
    
if __name__ == "__main__":
    main(verbose=True)
