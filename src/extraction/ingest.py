import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os
import torch
import argparse

from typing import Dict
from pathlib import Path
from ..config import BASE_OUT_DIR
from booknlp.booknlp import BookNLP
from ..auxiliary import preprocess, print_headers, print_information
from .booknlp_fix import exists_model_path, get_model_path, process_model_files


def init_run(file_name: str, out: str):
    """
    A convenience function executed as the initial run of the script
    to download all BERT models used by BookNLP. All subsequent executions
    of the script will run this method, but fail because of the `position_ids`
    that will then get removed.

    This setup prevents having to manually execute the initial run before
    being able to execute the actual script. The raised exception ensures
    that, once this function fails, the correct pipeline with the modified
    model parameters is loaded instead.
    """
    try:
        if exists_model_path():
            return

        model_params = {
            "pipeline": "entity,quote,supersense,event,coref",
            "model": "big",
        }

        # Run initial processing pipeline
        run_pipeline(file_name, model_params, out)
    except Exception as e:
        pass


def run_pipeline(file_name: str, model_params: Dict, out: str) -> None:
    """
    A convenience method to run the final `BookNLP` processing pipeline
    with the provided model paramaters on the given file.

    :param file_name: The name of the file to be processed
    :type file_name: str
    :param model_params: The dictionary containing the model parameters
    :type model_params: Dict
    """
    # Create the BookNLP pipeline object
    booknlp = BookNLP("en", model_params)

    # Input file to process
    input_file = file_name

    # Output directory to store resulting files in
    output_directory = out

    # File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
    book_id = f"{file_name.split('/')[-1].split('.')[0]}"

    # Run the processing pipeline
    booknlp.process(input_file, output_directory, book_id)


def main(file_name: Path, out: Path):
    """
    The main function of the script to execute the BookNLP pipeline.

    The modified models are loaded once their original version has been
    downloaded in `init_run`. Subsequent runs to the initial run fall
    through to the `finally`-block.
    """
    print_headers("BOOK INGESTION", "=")

    preproc_out_file = f"{out}/preproc_{file_name.name}"

    # Preprocess book file
    print_information(f"Preprocessing the file {file_name.name}", 1, "\n")
    preprocess(file_name, preproc_out_file)
    print_information(f"Preprocessed '{file_name.name}' by removing whitespace from sentences. Stored results in {preproc_out_file}", prefix="    ")

    print_information(f"Running BookNLP pipeline on '{preproc_out_file}'", 2, "\n")

    try:
        init_run(preproc_out_file, str(out))
    except Exception as e:
        pass
    finally:
        model_path = get_model_path()

        # Create custom model w/o original `position_ids` in BERT models
        model_params = {
            "pipeline": "entity,quote,supersense,event,coref",
            "model": "custom",
            "entity_model_path": str(
                model_path / "entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model"
            ),
            "coref_model_path": str(
                model_path / "coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model"
            ),
            "quote_attribution_model_path": str(
                model_path / "speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model"
            ),
            "bert_model_path": str(model_path.parent / ".cache/huggingface/hub/"),
            # "pronominalCorefOnly": False
            "referential_gender_cats": [
                ["he", "him", "his"],
                ["she", "her"],
                ["they", "them", "their"],
            ],
        }

        # Create the `torch.device`-Object
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Process the new model files with deleted `position_ids`
        model_params = process_model_files(model_params, device)

        # Run processing pipeline
        run_pipeline(preproc_out_file, model_params, out)
        print_information(f"Finished running BookNLP pipeline on '{file_name.name}'. Results stored in '{out}'", "✓", "\n", col="GREEN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to process a given input file in TXT-format using the BookNLP library."
    )

    parser.add_argument(
        "input_file", 
        type=Path, 
        help="The path to the input file to be processed"
        )
    
    parser.add_argument(
        "-o",
        "--out",
        default=BASE_OUT_DIR,
        help="The output directory to which the processing results are written",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
        
      # input_file.parent
    main(args.input_file, args.out)

    
