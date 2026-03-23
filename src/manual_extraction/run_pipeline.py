"""
CLI entry point for the 4-stage manual extraction pipeline.

Usage:
    python -m src.manual_extraction.run_pipeline [--stages 0,1,2,3,4] \\
        [--coref-model maverick|fastcoref] [--refine] \\
        [--include-non-protagonist] [-v] [-o OUT_DIR]
"""
from __future__ import annotations

import argparse
import time

from pathlib import Path
from .config import (
    RAW_TEXT, CLEAN_NAMES, OUT_DIR, 
    TOKENS, SPACY_MODEL, CHAR_RES_OUT, 
    COREF_OUT, VERB_OUT, OBJ_OUT
    )

from .utils import print_headers, print_information, load_spacy_model, suppress_stdout


def switch_venv():
    # Check for isolated coref environment
    coref_python = Path.cwd() / ".venv_coref" / "bin" / "python3"

    if coref_python.exists():
        print_information(f"Found isolated environment: {coref_python.parent.parent.name}", prefix="    ")

    return coref_python


def run_coreference(coref_python: str, args: argparse.Namespace, names_csv: Path, tokens_path: Path):
    import sys
    import subprocess

    cmd = [
        str(coref_python), "-m", "src.manual_extraction.coreference",
        "--text-path", str(args.text), # Added text_path
        "--char-res-dir", str(args.output / "char_resolution"),
        "--names-csv", str(names_csv), # Changed to names_csv
        "--out-dir", str(args.output),
        "--tokens-path", str(tokens_path),
        "--coref-model", "maverick"
    ]
    if args.refine:
        cmd.append("--refine")
    if args.verbose:
        cmd.append("-v")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print_information("Stage 2 (Isolated) failed.", "✗", col="RED")
        sys.exit(1)

    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4-Stage Manual Triple Extraction Pipeline",
    )
    parser.add_argument(
        "--stages", type=str, default="0,1,2,3,4",
        help="Comma-separated list of stages to run (0=tokenise, 1-4=pipeline). Default: 0,1,2,3,4",
    )
    parser.add_argument(
        "--coref-model", type=str, default="maverick",
        choices=["maverick", "fastcoref"],
        help="Coreference model to use. Default: maverick",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="Enable iterative Stage 1 ↔ 2 refinement pass",
    )
    parser.add_argument(
        "--include-non-protagonist", action="store_true",
        help="Also extract triples where the object is not a protagonist",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=OUT_DIR,
        help=f"Output directory. Default: {OUT_DIR}",
    )
    parser.add_argument(
        "--text", type=Path, default=RAW_TEXT,
        help=f"Path to input text file. Default: {RAW_TEXT}",
    )
    parser.add_argument(
        "--names", type=Path, default=CLEAN_NAMES,
        help=f"Path to canonical names CSV. Default: {CLEAN_NAMES}",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ignore warnings
    from warnings import simplefilter
    simplefilter("ignore")

    # Parse stages
    stages = [int(s.strip()) for s in args.stages.split(",")]

    # Resolve paths
    out_dir = Path(args.output) if args.output else OUT_DIR
    text_path = Path(args.text) if args.text else RAW_TEXT
    names_csv = Path(args.names) if args.names else CLEAN_NAMES

    out_dir.mkdir(parents=True, exist_ok=True)

    print_headers("4-STAGE MANUAL EXTRACTION PIPELINE", "=", prefix="\n")
    print_information(f"Stages to run: {stages}", prefix="    ")
    print_information(f"Input text: {text_path}", prefix="    ")
    print_information(f"Names CSV: {names_csv}", prefix="    ")
    print_information(f"Output dir: {out_dir}", prefix="    ")
    if 2 in stages:
        print_information(f"Coref model: {args.coref_model}", prefix="    ")
    if args.refine:
        print_information("Iterative refinement: ENABLED", prefix="    ")
    if args.include_non_protagonist:
        print_information("Non-protagonist triples: ENABLED", prefix="    ")
    print()

    overall_start = time.time()

    # Pre-load heavy models to avoid repeatedly loading them in each stage
    global_nlp = None
    global_gliner = None
    
    stages_set = set(stages)
    if stages_set.intersection({0, 1, 3, 4}):
        print_information(f"Pre-loading global spaCy model '{SPACY_MODEL}'...", prefix="")
        global_nlp = load_spacy_model()
        # global_nlp = spacy.load(SPACY_MODEL)
    
    if stages_set.intersection({1}) or (4 in stages_set and args.include_non_protagonist):
        print_information("Pre-loading global GLiNER model...", prefix="")
        from gliner import GLiNER
        with suppress_stdout():
            global_gliner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        
    print()

    if text_path == RAW_TEXT and out_dir == OUT_DIR:
        tokens_path = TOKENS
    else:
        tokens_path = out_dir / text_path.name.replace(".md", ".tokens")

    # ============================
    # Stage 0 — Tokenisation
    # ============================
    if 0 in stages:
        t0 = time.time()
        from .tokenise import main as tokenise
        tokens_path = tokenise(text_path=text_path, out_dir=out_dir, verbose=args.verbose, nlp=global_nlp)
        print_information(f"Stage 0 completed in {time.time() - t0:.1f}s\n", "✓", col="GREEN")

    # ============================
    # Stage 1 — NER + Fuzzy Match
    # ============================
    if 1 in stages:
        t0 = time.time()
        from .character_resolution import main as character_resolution
        character_resolution(
            text_path=text_path,
            names_csv=names_csv,
            out_dir=out_dir,
            tokens_path=tokens_path,
            verbose=args.verbose,
            nlp=global_nlp,
            gliner_model=global_gliner,
        )
        print_information(f"Stage 1 completed in {time.time() - t0:.1f}s\n", "✓", col="GREEN")

    # ============================
    # Stage 2 — Coreference
    # ============================
    if 2 in stages:
        t0 = time.time()
        print_information("Launching Stage 2...", prefix="")

        # Check for isolated coref environment.
        # Required due to dependency conflicts between gliner and maverick.
        if args.coref_model == "maverick":
            coref_python = switch_venv()
            run_coreference(coref_python, args, names_csv, tokens_path)              

        # from .coreference import main as coreference
        # coreference(
        #     text_path=text_path,
        #     stage1_dir=out_dir / "stage1",
        #     names_csv=names_csv,
        #     out_dir=out_dir,
        #     coref_model_name=args.coref_model,
        #     refine=args.refine,
        #     verbose=args.verbose,
        # )

        print_information(f"Stage 2 completed in {time.time() - t0:.1f}s\n", "✓", col="GREEN")

    # ============================
    # Stage 3 — Verb Extraction
    # ============================
    if 3 in stages:
        t0 = time.time()
        from .verb_extraction import main as verb_extraction
        verb_extraction(
            tokens_path=tokens_path,
            coref_dir=out_dir / "coreference",
            out_dir=out_dir,
            verbose=args.verbose,
            nlp=global_nlp,
        )
        print_information(f"Stage 3 completed in {time.time() - t0:.1f}s\n", "✓", col="GREEN")

    # ============================
    # Stage 4 — Object Extraction
    # ============================
    if 4 in stages:
        t0 = time.time()
        from .object_extraction import main as object_extraction
        object_extraction(
            text_path=text_path,
            tokens_path=tokens_path,
            coref_dir=out_dir / "coreference",
            verb_ext_dir=out_dir / "verb_extraction",
            names_csv=names_csv,
            out_dir=out_dir,
            include_non_protagonist=args.include_non_protagonist,
            verbose=args.verbose,
            nlp=global_nlp,
            gliner_model=global_gliner,
        )
        print_information(f"Stage 4 completed in {time.time() - t0:.1f}s\n", "✓", col="GREEN")

    # Done
    elapsed = time.time() - overall_start
    print_headers("PIPELINE COMPLETE", "=", prefix="\n")
    print_information(f"Total time: {elapsed:.1f}s", "✓", col="GREEN")
    print_information(f"Outputs in: {out_dir}", prefix="    ")


if __name__ == "__main__":
    main()
