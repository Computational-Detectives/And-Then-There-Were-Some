# And Then There Were Some

Computational Network Analysis to spot the killer in the famous novel "And Then There Were None" by Agatha Christie. 🕵️💻

Stay tuned!

## Installation
We recommend to follow the below steps on a Unix machine to reproduce the environment used for this project. Importantly, because the `Torch` library for now still depends on Python `<=3.12` it is necessary to use a Python version not higher than `3.12`.

```bash
# Create & activate virtual environment on Unix (or Windows equivalent)
python3.12 -m venv <venv_name>
source <venv_name>/bin/activate

# Install packages
pip install -r requirements.txt

# Download English language processing model
python -m spacy download en_core_web_sm
```

## Run Extraction Pipeline
To run the whole extraction pipeline, run the script as shown below. This will store the output in `out/` under the condition that all issues with `BookNLP` have been solved prior to execution. A fix that has been tested to work on Unix/macOS systems is located in `booknlp_fix.py`.

```bash
# Run script (can be run as-is, no arguments need to be passed)
python run_all.py

# Display usage information
python run_all.py -h
usage: run_all.py [-h] [-i INPUT] [-o OUT] [-v]

Run the whole end-to-end processing pipeline

options:
  -h, --help               show this help message and exit
  -i INPUT, --input INPUT  The path to the input file to be processed
  -o OUT, --out OUT        The output directory to which the processing results are written
  -v, --verbose            A flag to trigger verbose output
```

*Note: Each file executed in `run_all.py` can also be run individually and may, depending on the file, provide more granular control over the results produced. For example, running `extract_avp_triples.py` with the `-t`-argument allows for filtering triples based on token ranges. In this way the analysis can be focused on individual chapters of the book.*
