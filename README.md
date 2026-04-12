# And Then There Were Some

Computational Network Analysis to spot the killer in the famous novel "And Then There Were None" by Agatha Christie. 🕵️💻


## Installation
We recommend to follow the below steps on a Unix machine to reproduce the environment used for this project. Importantly, because the `Torch` library for now still depends on Python `<=3.12` it is necessary to use a Python version not higher than `3.12`.

Furthermore, because of some libraries (`GLiNER` & `Maverick`) depending one different versions of the same parent library, two virtual Python environments have to be set up to prevent them from interfering with each other.

```bash
# 1) Create 1st & main virtual environment on Unix (or Windows equivalent)
# The name for the first environment does not matter
python3.12 -m venv <venv_name>

# 2) Activate the main VENV
source <venv_name>/bin/activate

# 3) Install packages
pip install -r requirements.txt

# 4) Download English language processing model
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('verbnet'); nltk.download('framenet_v17'); nltk.download('wordnet')"

# 5) Repeat step (1) & (2) and name the 2nd VENV '.coref_venv'. 
# 6) Install '.coref_venv'-specific packages
pip install -r requirements_coref.txt
```

## Run Extraction Pipeline
To run the whole extraction pipeline from the project root with the main VENV activated as shown below. This will store the results in `out/`.

```bash
# Run script (can be run as-is, no arguments need to be passed)
python run_pipeline.py

# Display usage information
python run_pipeline.py -h
usage: run_pipeline.py [-h] [--text TEXT] [-o OUTPUT] [--stages STAGES] [--refine] [--coref-model {maverick}] [--include-non-protagonist] [--names NAMES]
                       [-v] [--debug]

4-Stage Manual Triple Extraction Pipeline

options:
  -h, --help            show this help message and exit
  --text TEXT           Path to input text file. Default: /data/book/attwn.md
  -o OUTPUT, --output OUTPUT
                        Output directory. Default: /out
  --stages STAGES       Comma-separated list of stages to run (0=tokenise, 1-4=pipeline). Default: 0,1,2,3,4
  --refine              Enable iterative Stage 1 ↔ 2 refinement pass
  --coref-model {maverick}
                        Coreference model to use. Default: maverick
  --include-non-protagonist
                        Also extract triples where the object is not a protagonist
  --names NAMES         Path to canonical names CSV. Default: /data/names_owen_split.csv
  -v, --verbose         Verbose output
  --debug               Start the spaCy model server for faster iterative runs. (Currently doesn't do anything.)
```

*Note: Each stage in `run_pipeline.py` can also be run individually using the `--stages [STAGES]` argument.*


### Visualisation & Verification of Triples
The triples produced as a result of running of running the extraction pipeline (by default stored in `/out/obj_extraction/triples.csv`) can be visualised in the browser using the below script. The extracted triples can be split into chunks for sequential or distributed manual processing. Changes to the triples can be applied locally and downloaded in the `TSV` file format for further processing via `process_corrections.py`.


#### Visualisation of Triples
```bash
# Create visualisation of extracted triples. Run as Python module from main VENV.
python -m src.verfication.verify_triples -h
usage: verify_triples.py [-h] [--text TEXT] [--tokens TOKENS] [--triples TRIPLES] [--coref-dir COREF_DIR] [--out-dir OUT_DIR] [--out-html OUT_HTML]
                         [--split SPLIT]

Generate Triple Verification HTML

options:
  -h, --help            show this help message and exit
  --text TEXT           Path to raw text file
  --tokens TOKENS       Path to .tokens file
  --triples TRIPLES     Path to triples.csv
  --coref-dir COREF_DIR
                        Directory containing span_index.jsonl
  --out-dir OUT_DIR     Output directory
  --out-html OUT_HTML   Output HTML file path
  --split SPLIT         Number of splits to divide triples into (default: 1 = no split)
```

#### Processing of Corrections
```bash
# Process corrections by running the following script as a Python module from the main VENV
python -m src.verification.process_corrections -h
usage: process_corrections.py [-h] [--triples TRIPLES] [--tokens TOKENS] [-o OUT] [--names NAMES] [-v]

Visualize extracted triples and allow for corrections to be made to them

options:
  -h, --help         show this help message and exit
  --triples TRIPLES  Path to corrected triples
  --tokens TOKENS    Path to extracted tokens
  -o OUT, --out OUT  Output directory. Default: /out
  --names NAMES      Path to canonical names CSV. Default: /data/names_owen_split.csv
  -v, --verbose      Verbose output
```

## Run Analysis Pipeline
To reproduce the results obtained in the study, run `run_analysis.py`.

```bash
# Run script (can be run as-is, no arguments need to be passed)
python run_pipeline.py

# Display usage information
python run_pipeline.py -h
usage: run_pipeline.py [-h] [--text TEXT] [-o OUTPUT] [--stages STAGES] [--refine] [--coref-model {maverick}] [--include-non-protagonist] [--names NAMES]
                       [-v] [--debug]

4-Stage Manual Triple Extraction Pipeline

options:
  -h, --help            show this help message and exit
  --text TEXT           Path to input text file. Default: /data/book/attwn.md
  -o OUTPUT, --output OUTPUT
                        Output directory. Default: /out
  --stages STAGES       Comma-separated list of stages to run (0=tokenise, 1-4=pipeline). Default: 0,1,2,3,4
  --refine              Enable iterative Stage 1 ↔ 2 refinement pass
  --coref-model {maverick}
                        Coreference model to use. Default: maverick
  --include-non-protagonist
                        Also extract triples where the object is not a protagonist
  --names NAMES         Path to canonical names CSV. Default: /data/names_owen_split.csv
  -v, --verbose         Verbose output
  --debug               Start the spaCy model server for faster iterative runs. (Currently doesn't do anything.)
```
