from __future__ import annotations

import sys
import time
import argparse

from pathlib import Path
from datetime import datetime
from src.config import (
    RAW_TEXT, CLEAN_NAMES, 
    OUT_DIR, TOKENS, SPACY_MODEL
    )

from src.extraction.utils import (
    print_headers, print_information, 
    load_spacy_model, load_gliner_model,
    suppress_stdout, setup_pipeline_logger
    )


# ============================================================
# ---------------------- MODEL SERVER ------------------------
# ============================================================
MODEL_SERVER_URL = "http://127.0.0.1:8765"

def _server_is_up() -> bool:
    """Check if the model server is responding."""
    try:
        import requests
        resp = requests.get(f"{MODEL_SERVER_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_model_server():
    """
    Start the model server if it isn't already running.
    Blocks until the server is healthy or raises after timeout.
    """
    import subprocess

    if _server_is_up():
        print_information("Model server already running.\n", prefix="    ")
        return

    print_information("Starting model server...", prefix="    ")
    subprocess.Popen(
        [sys.executable, "model_server.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for it to come up
    for _ in range(240):
        time.sleep(1)
        if _server_is_up():
            print_information("Model server ready.", "✓", col="GREEN")
            return

    raise RuntimeError("Model server failed to start within 30 seconds.")


# ============================================================
# -------------------- COREF VENV SWITCH ---------------------
# ============================================================
def switch_venv():
    # Check for isolated coref environment
    coref_python = Path.cwd() / ".venv_coref" / "bin" / "python3"

    if coref_python.exists():
        print_information(f"Found isolated environment: {coref_python.parent.parent.name}", prefix="    ")

    return coref_python


def run_coreference(coref_python: str, args: argparse.Namespace, names_csv: Path, tokens_path: Path):
    import subprocess

    cmd = [
        str(coref_python), "-m", "src.extraction.coreference",
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


def main(args: argparse.Namespace):
    # Ignore warnings
    from warnings import simplefilter
    simplefilter("ignore")

    # Start timestamp
    start_time = datetime.now()

    # Parse stages
    stages = [int(s.strip()) for s in args.stages.split(",")]

    # Resolve paths
    out_dir = Path(args.output)
    text_path = Path(args.text)
    names_csv = Path(args.names)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure pipeline logger → file
    # import logging
    # pipeline_logger = logging.getLogger("pipeline")
    # pipeline_logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    # # fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    # fh.setFormatter(logging.Formatter("%(message)s"))
    # pipeline_logger.addHandler(fh)

    log_path = out_dir / "pipeline.log"
    setup_pipeline_logger(log_path=log_path)

    # ============================================================
    # ---------------------- PRINT PREAMBLE ----------------------
    # ============================================================
    print_headers("4-STAGE MANUAL EXTRACTION PIPELINE", "=", prefix="\n")
    print_information(f"Pipeline started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
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

    # =============================================================
    # ---------------------- PRE-LOAD MODELS ----------------------
    # =============================================================
    # Pre-load heavy models to avoid repeatedly loading them in each stage
    global_nlp = None
    global_gliner = None
    
    stages_set = set(stages)
    if stages_set.intersection({0, 1, 3, 4}):
        if args.debug:
            print_information("Debug mode: ensuring model server is running...", symb="!", prefix="", col="YELLOW")
            ensure_model_server()
            print_information(f"Using spaCy model server via proxy '{MODEL_SERVER_URL}'...", symb='*', prefix="")
        else:
            print_information(f"Pre-loading global spaCy model '{SPACY_MODEL}' locally...", symb='*', prefix="")
        global_nlp = load_spacy_model(
            server_url=MODEL_SERVER_URL if args.debug else None
        )
    
    if stages_set.intersection({1}) or (4 in stages_set and args.include_non_protagonist):
        print_information("Pre-loading global GLiNER model...", symb='*', prefix="")
        with suppress_stdout():
            global_gliner = load_gliner_model()
        
    print()

    # =============================================================
    # ----------------------- TOKENIZATION ------------------------
    # =============================================================
    tokens_path = (TOKENS 
                   if text_path == RAW_TEXT and out_dir == OUT_DIR 
                   else out_dir / text_path.name.replace(".md", ".tokens"))
    
    if 0 in stages:
        t0 = time.time()
        from src.extraction.tokenise import main as tokenise
        tokens_path = tokenise(text_path=text_path, out_dir=out_dir, verbose=args.verbose, nlp=global_nlp)
        print_information(f"Stage 0 completed in {time.time() - t0:.1f}s\n", "✓", col="GREEN")


    # =============================================================
    # --------------------- NER + FUZZY MATCH ---------------------
    # =============================================================
    if 1 in stages:
        t0 = time.time()
        from src.extraction.character_resolution import main as character_resolution
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


    # =============================================================
    # ------------------------ COREFERENCE ------------------------
    # =============================================================
    if 2 in stages:
        t0 = time.time()
        print_information("Launching Stage 2...", prefix="")

        # Check for isolated coref environment.
        # Required due to dependency conflicts between gliner and maverick.
        if args.coref_model == "maverick":
            coref_python = switch_venv()
            run_coreference(coref_python, args, names_csv, tokens_path)              

        print_information(f"Stage 2 completed in {time.time() - t0:.1f}s\n", "✓", col="GREEN")


    # =============================================================
    # ---------------------- VERB EXTRACTION ----------------------
    # =============================================================
    if 3 in stages:
        t0 = time.time()
        from src.extraction.verb_extraction import main as verb_extraction
        verb_extraction(
            tokens_path=tokens_path,
            coref_dir=out_dir / "coreference",
            out_dir=out_dir,
            verbose=args.verbose,
            nlp=global_nlp,
        )
        print_information(f"Stage 3 completed in {time.time() - t0:.1f}s\n", "✓", col="GREEN")


    # =============================================================
    # --------------------- OBJECT EXTRACTION ---------------------
    # =============================================================
    if 4 in stages:
        t0 = time.time()
        from src.extraction.object_extraction import main as object_extraction
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

    # --------------------- DONE ---------------------
    end_time = datetime.now()
    duration = end_time - start_time
    elapsed = time.time() - overall_start
    print_headers("PIPELINE COMPLETE", "=", prefix="\n")
    print_information(f"Pipeline finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (duration: {duration})")
    print_information(f"Total time: {elapsed:.1f}s", "✓", col="GREEN")
    print_information(f"Outputs in: {out_dir}", prefix="    ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="4-Stage Manual Triple Extraction Pipeline",
    )
    
    parser.add_argument(
        "--text", type=Path, default=RAW_TEXT,
        help=f"Path to input text file. Default: {RAW_TEXT}",
    )
    
    parser.add_argument(
        "-o", "--output", type=Path, default=OUT_DIR,
        help=f"Output directory. Default: {OUT_DIR}",
    )
    
    parser.add_argument(
        "--stages", type=str, default="0,1,2,3,4",
        help="Comma-separated list of stages to run (0=tokenise, 1-4=pipeline). Default: 0,1,2,3,4",
    )
    
    parser.add_argument(
        "--refine", action="store_true",
        help="Enable iterative Stage 1 ↔ 2 refinement pass",
    )
    
    parser.add_argument(
        "--coref-model", type=str, default="maverick",
        choices=["maverick"], # , "fastcoref"],
        help="Coreference model to use. Default: maverick",
    )
    
    parser.add_argument(
        "--include-non-protagonist", action="store_true",
        help="Also extract triples where the object is not a protagonist",
    )
    
    parser.add_argument(
        "--names", type=Path, default=CLEAN_NAMES,
        help=f"Path to canonical names CSV. Default: {CLEAN_NAMES}",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--debug", action="store_true",
        help="Start the spaCy model server for faster iterative runs.",
    )

    args = parser.parse_args()

    main(args=args)
