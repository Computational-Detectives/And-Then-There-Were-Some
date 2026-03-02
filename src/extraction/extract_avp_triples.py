import os
import argparse
import pandas as pd


from pathlib import Path
from ..config import TOKENS, BASE_OUT_DIR, TRIPLE_OUT
from .extract_svo_triples import make_doc_from_sentence, is_negated
from ..auxiliary import get_chapter_token_range, int_range, print_headers, print_information


def get_negation(avp: pd.DataFrame, tokens_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each AVP triple, reconstruct the sentence and check if the verb is negated.
    """
    from spacy.tokens import Token
    Token.set_extension("global_id", default=None, force=True)
    
    # Initialize negated column
    avp['negated'] = False
    
    # Process by sentence to avoid reconstructing the same Doc object multiple times
    sent_ids = avp['sentence_ID'].dropna().unique()
    
    for sent_id in sent_ids:
        # Get all tokens for this sentence
        sentence_tokens = tokens_df[tokens_df['sentence_ID'] == sent_id].reset_index(drop=True)
        if sentence_tokens.empty:
            continue
        
        # Skip sentences with invalid syntactic_head_ID values
        try:
            doc = make_doc_from_sentence(sentence_tokens)
        except (ValueError, TypeError):
            # Skip sentences that can't be parsed (e.g., non-numeric syntactic_head_ID)
            continue
        
        # Build index for tokens in doc using the global_id extension
        token_map = {token._.global_id: token for token in doc}
        
        # Find which AVP triples belong to this sentence
        mask = avp['sentence_ID'] == sent_id
        for idx, row in avp[mask].iterrows():
            verb_idx = row['index']
            token = token_map.get(verb_idx)
            
            if token:
                # Use is_negated to check negation children
                avp.at[idx, 'negated'] = is_negated(token)
                
    return avp


def extract_avp(
        characters: pd.DataFrame, 
        out: Path = TRIPLE_OUT, 
        tokens_path: str = TOKENS, 
        verbose: bool = False
    ):
    """
    Extracts Agient-Verb-Patient triples from information extracted by running
    the `BookNLP` pipeline with its results stored in `.book`.

    The function removes self-loops and extracts information on whether the verb
    in the interaction is negated or not.
    
    :param characters: A DataFrame that contains information on each character's interaction
    :type characters: pd.DataFrame
    :param out: The directory to write the AVP triples to
    :type out: Path
    :param tokens_path: The path to the token file for negation extraction
    :type tokens_path: str
    :param verbose: A flag to print verbose statistics
    :type verbose: bool
    """
    print_headers("AGENT-VERB-PATIENT TRIPLE EXTRACTION", "=", prefix="\n")

    # Load tokens from the provided path
    tokens_df = pd.read_csv(tokens_path, sep="\t")

    print_information("Merging to extract AVP triples...", 1, "\n")

    # Get triples by merging on index. Creates all combinations of triples for the same index.
    merged = characters.merge(
        characters,
        on='index',
        suffixes=['_left', '_right']
    )

    # Removes self-loops
    avp = merged[
        merged["canonical_id_left"] != merged["canonical_id_right"]
    ]

    # Remove all agent-agent and patient-patient rows.
    # Self-join already ensures the corresponding triples exist.
    same_role = (avp['role_left'] == avp['role_right'])
    avp = avp[~same_role]

    # Mirror rows so the agent is always on the left
    pa_mask = (avp['role_left'] == 'patient') & (avp['role_right'] == 'agent')
    left_cols  = [c for c in avp.columns if c.endswith('_left')]
    right_cols = [c.replace('_left', '_right') for c in left_cols]
    for lc, rc in zip(left_cols, right_cols):
        avp.loc[pa_mask, [lc, rc]] = avp.loc[pa_mask, [rc, lc]].values

    # Deduplicate after mirroring as
    # patient -> agent rows overlap with agent -> patient counterparts
    avp = avp.drop_duplicates(
        subset=['canonical_id_left', 'canonical_id_right', 'index'],
        keep='first'
    ).reset_index(drop=True)

    # Merge avp with tokens to get the lemma for word_agent
    # Match avp['index'] with tokens['token_ID_within_document']
    avp = avp.merge(
        tokens_df[['token_ID_within_document', 'lemma', 'sentence_ID']], 
        left_on='index', 
        right_on='token_ID_within_document',
        how='left'
    )
        
    # Determine negation status for each triple
    avp = get_negation(avp, tokens_df)
    
    # Drop the temporary columns we don't need
    avp['source'] = 'avp'
    avp['word'] = avp['word_left']

    avp = avp[['source', 'canonical_id_left', 'name_left', 'role_left',
       'word', 'lemma', 'index', 'negated', 'original_ids_left', 'gender_left',
       'name_variants_left', 'canonical_id_right',
       'name_right', 'role_right', 'original_ids_right',
       'gender_right', 'name_variants_right']]
    
    # Collect simple statistics
    total_entries = merged.shape[0]
    num_triples = avp.shape[0]
    dropped = total_entries - num_triples
    num_negated = avp['negated'].sum()
    
    print_information(f"Extracted {num_triples} AVP triples", prefix="    ")

    if verbose:
        print_headers("AVP EXTRACTION SUMMARY", "-", prefix="\n")

        print(f"    Total entries:          {total_entries}")
        print(f"    Triples created:        {num_triples} ({num_triples / total_entries * 100:.1f}%)")
        print(f"    Entries dropped:        {dropped} ({dropped / total_entries * 100:.1f}%)")
        print(f"    Negated triples:        {num_negated} ({num_negated / num_triples * 100:.1f}%)")

        print("\nNOTE: These statistics are potentially misleading as triple creation cannot necessarily be expressed as a percentage of total entries.")

    if not os.path.isdir(out):
        os.makedirs(out)
        
    avp.to_csv(f"{out}/avp_triples.csv", sep="\t", index=False)
    print_information(f"AVP triples saved to: {out}", "✓", prefix="\n", col="GREEN")


if __name__ == '__main__':
    description = f"Use this script to create the initial network of characters\nFollowing token ranges are available for purposes of filtering:\n\n{get_chapter_token_range()}"
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=Path(BASE_OUT_DIR + '/merged_characters.characters'),
        help='Specify the input file to use for triple extraction'
    )

    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(BASE_OUT_DIR),
        help="The output directory which to save the graphs to",
    )
    
    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Path to the tokens file. If not provided, derives from input file's parent directory."
    )
    
    parser.add_argument(
        "-t",
        "--token-range",
        default=[],
        type=int_range,
        help="A comma-seperated interval '[x, y]' of token IDs for which to perform graph-creation",
    )

    parser.add_argument(
        '-v', '--verbose', 
        action="store_true",
        help="A flag to trigger verbose output"
    )

    args = parser.parse_args()

    characters = pd.read_csv(args.input, sep="\t")

    # Derive tokens path from input file's parent directory if not explicitly provided
    if args.tokens:
        tokens_path = args.tokens
    else:
        tokens_path = str(args.input.parent / "preproc_attwn.tokens")

    if args.token_range:
        # If a token interval is provided, filter the characters DF
        # based on the range for further analysis
        characters = characters[
            (characters["index"] >= args.token_range[0])
            & (characters["index"] <= args.token_range[1])
        ]

    extract_avp(characters, args.out, tokens_path=tokens_path, verbose=args.verbose)

