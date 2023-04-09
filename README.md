# Baseline TF-IDF Analysis (WP3)

## Introduction

The Baseline TF-IDF Analysis (WP3) provides baseline scoring using conventional machine learning classifiers and methods.

## Installation

`pip install -r requirements.txt` 

## Recommended Requirements

- CPUs: 10 (based on n_jobs setting in classification.py)
- RAM: 256gb

## Configuration

the configuration file, `config.py`, can be used in order to set:

- `MATERIAL_PATH`:  The path for inputs (e.g., input data, intermediary data sets).
- `RESULTS_PATH`:  The path for output (e.g., tables, images).
- `RANDOM_STATE`: This setting will be used as an argument to all random_state parameters.  This setting can be hard-coded in order to ensure reproducibility.  

## Usage

The main entry point for the pipeline is the file named `main.py`.  `Main.py` will export the notebooks to `.py` files and run each step of the pipeline. The pipeline may take several days to run.

Run the pipeline:

```bash
python main.py
```

##  Data Access Statement

### Ensembl Data

"Ensembl imposes no restrictions on access to, or use of, the data provided and the software used to analyse and present it. Ensembl data generated by members of the project are available without restriction. Ensembl code written by members of the project is provided under the Apache 2.0 licence.

Some of the data and software included in the distribution may be subject to third-party constraints. Users of the data and software are solely responsible for establishing the nature of and complying with any such restrictions.

The European Molecular Biology Laboratory's European Bioinformatics Institute (EMBL-EBI) provides this data and software in good faith, but make no warranty, express or implied, nor assume any legal liability or responsibility for any purpose for which they are used (https://useast.ensembl.org/info/about/legal/disclaimer.html)."

### Uniprot Data

"We have chosen to apply the Creative Commons Attribution 4.0 International (CC BY 4.0) License) to all copyrightable parts of our databases (https://www.uniprot.org/help/license)."

### IMPC Data

"The IMPC makes this dataset publicly available under the Creative Commons Attribution 4.0 International license (CC-BY 4.0) (https://www.mousephenotype.org/help/faqs/is-impc-data-freely-available/)."

### Works generated by the pipeline.

Works generated by this pipeline are licensed under the Attribution 4.0 International (CC BY 4.0) (https://creativecommons.org/licenses/by/4.0/legalcode).
