import re
import argparse

from pathlib import Path
from src.config import CLEAN_NAMES, OUT_DIR


INVISIBLE_CHARS = re.compile(r'[\u200B-\u200F\u2060\uFEFF]')

def clean_str(x):
    if not isinstance(x, str):
        return x
    return INVISIBLE_CHARS.sub('', x).strip()


def main(args):
    try:
        import pandas as pd
        # Load & clean triples
        df = pd.read_csv(args.triples, sep='\t', dtype=str, keep_default_na=False)
        df = df.map(clean_str)

        tokens_df = pd.read_csv(args.tokens, sep='\t', dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    # Convert numeric fields in tokens_df for efficient querying
    tokens_df['byte_onset'] = pd.to_numeric(tokens_df['byte_onset'], errors='coerce')
    tokens_df['byte_offset'] = pd.to_numeric(tokens_df['byte_offset'], errors='coerce')

    # Load canonical ids from CLEAN_NAMES
    name_to_id = {}
    try:
        names_df = pd.read_csv(CLEAN_NAMES, dtype=str, keep_default_na=False)
        for _, r in names_df.iterrows():
            cid = str(r.get('id', '')).strip()
            fname = str(r.get('fullname', '')).strip()
            if fname:
                name_to_id[fname] = cid
            aka_str = str(r.get('aka', '')).strip()
            if aka_str:
                for a in aka_str.split(';'):
                    a = a.strip()
                    if a:
                        name_to_id[a] = cid
    except Exception as e:
        print(f"Error loading {CLEAN_NAMES}: {e}")

    output_headers = [
        "canonical_id_left", "name_left", "role_left", "word", "lemma", "index",
        "negated", "gender_left", "name_variants_left", "canonical_id_right",
        "name_right", "role_right", "gender_right", "name_variants_right", "is_new"
    ]
    
    # Collect statistics
    num_triples = df.shape[0]
    stats = {
        "total_num_triples_start": num_triples,
        "invalid_triples_total": {'count': 0, '%': 0},
        "invalid_triples_agent": {'count': 0, '%': 0},
        "invalid_triples_patient": {'count': 0, '%': 0},
        "total_triples_corrected": {'count': 0, '%': 0},
        "total_corrections": {'count': 0, '%': 0},
        "changes_agent": {'count': 0, '%': 0},
        "changes_verb": {'count': 0, '%': 0},
        "changes_patient": {'count': 0, '%': 0},
        "correct_triples": {'count': 0, '%': 0},
        "new_triples": {'count': 0, '%': 0},
    }
    
    output_rows = []
    
    for idx, row in df.iterrows():
        # Determine whether a triple is changed and/or new
        is_changed = str(row.get('changed', '')).strip().lower() == 'true'
        is_new = str(row.get('is_new', '')).strip().lower() == 'true'           

        # Convert row to dict
        out_row = {h: row.get(h, "") for h in output_headers}
        
        # Initialize the variant columns to empty string if not in source
        for col in ["name_variants_left", "name_variants_right"]:
            if col not in row:
                out_row[col] = ""

        if is_changed:
            # Check if 'INCORRECT_TRIPLE' (invalid) is in corrected_{agent, patient}
            # Invalid triples are thrown away because they are invalid 
            corr_agent = str(row.get('corrected_agent', '')).strip()
            corr_patient = str(row.get('corrected_patient', '')).strip()

            # Check if is invalid
            invalid_agent = corr_agent == 'INCORRECT_TRIPLE'
            invalid_patient = corr_patient == 'INCORRECT_TRIPLE'

            if invalid_agent or invalid_patient:
                stats["invalid_triples_total"]['count'] += 1
                if invalid_agent:
                    stats["invalid_triples_agent"]['count'] += 1
                if invalid_patient:
                    stats["invalid_triples_patient"]['count'] += 1
                continue
            
            change_applied = str(row.get('changes_applied')).strip()
            # 
            if any(x in change_applied for x in ['agent', 'verb', 'patient']):
                stats["total_triples_corrected"]['count'] += 1

            if 'agent' in change_applied:
                stats["changes_agent"]['count'] += 1
            
            if 'verb' in change_applied:
                stats["changes_verb"]['count'] += 1

            if 'patient' in change_applied:
                stats["changes_patient"]['count'] += 1

            if corr_agent:                
                # Overwrite name_left with the corrected agent value
                out_row['name_left'] = corr_agent
            
            if row.get('corrected_verb', '').strip():
                out_row['word'] = str(row.get('corrected_verb')).strip()
                
            if corr_patient:
                out_row['name_right'] = corr_patient
            
        if is_new:
            # Count number of new triples
            stats["new_triples"]['count'] += 1

            v_start = row.get('verb_byte_start', '')
            v_end = row.get('verb_byte_end', '')
            
            if v_start and v_end and str(v_start).isdigit() and str(v_end).isdigit():
                v_start = int(v_start)
                v_end = int(v_end)
                
                # Overlap condition to find token
                matches = tokens_df[(tokens_df['byte_onset'] < v_end) & (tokens_df['byte_offset'] > v_start)]
                if len(matches) > 0:
                    verb_matches = matches[matches['POS_tag'].str.startswith('V')]
                    if len(verb_matches) > 0:
                        token = verb_matches.iloc[0]
                    else:
                        token = matches.iloc[0]
                    
                    out_row['index'] =  str(token['token_ID_within_document'])
                    out_row['lemma'] = str(token.get('lemma', ''))
                    
                    sent_id = token.get('sentence_ID')
                    doc_tok_id = token.get('token_ID_within_document')
                    
                    sent_tokens = tokens_df[tokens_df['sentence_ID'] == sent_id]
                    neg_tokens = sent_tokens[(sent_tokens['dependency_relation'] == 'neg') & (sent_tokens['syntactic_head_ID'] == str(doc_tok_id))]
                    
                    if len(neg_tokens) > 0:
                        out_row['negated'] = "true"
                    else:
                        out_row['negated'] = "false"            

        if not is_changed and not is_new:
            stats["correct_triples"]['count'] += 1

        # Lookup canonical IDs
        nl = str(out_row.get('name_left', '')).strip()
        nr = str(out_row.get('name_right', '')).strip()
        
        if nl in name_to_id:
            out_row['canonical_id_left'] = name_to_id[nl]
        if nr in name_to_id:
            out_row['canonical_id_right'] = name_to_id[nr]

        output_rows.append(out_row)

    stats['total_corrections']['count'] += stats['changes_agent']['count'] + stats['changes_verb']['count'] + stats['changes_patient']['count']

    for i, k in enumerate(stats):
        if i == 0:
            continue
        stats[k]['%'] = f"{stats[k]['count'] / (num_triples - stats['new_triples']['count']) * 100:.2f}"

    # Print & save statistics of processing
    out_dir = Path(args.out) / "obj_extraction" / "corrections" / "processed"
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['count | %'])
    stats_df.index.name = "Stats"

    stats_df.to_csv(out_dir / "processing_stats_updated.csv", sep='\t', index=True, index_label='Stats')

    print("\n=== Processing Statistics ===")
    print(stats_df)
    print()

    out_df = pd.DataFrame(output_rows, columns=output_headers)
    out_df.to_csv(out_dir / 'processed_triples_updated.csv', sep='\t', index=False)
    print(f"Processed {len(df)} lines into {len(out_df)} lines.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize extracted triples and allow for corrections to be made to them")

    parser.add_argument(
        "--triples", type=Path,
        help="Path to corrected triples",
    )
    
    parser.add_argument(
        "--tokens", type=Path,
        help="Path to extracted tokens",
    )

    parser.add_argument(
        "-o", "--out", type=Path, default=OUT_DIR,
        help=f"Output directory. Default: {OUT_DIR}",
    )
    
    parser.add_argument(
        "--names", type=Path, default=CLEAN_NAMES,
        help=f"Path to canonical names CSV. Default: {CLEAN_NAMES}",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    main(args)
