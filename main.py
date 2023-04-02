# %%
import os as os
import sys as sys
import re as re
import pandas as pd
import numpy as np
import json as json
from subprocess import run

pd.options.display.max_columns = 100
pd.options.display.min_rows = None
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 100

from config import RANDOM_STATE, RESULTS_PATH

# %%
run(f"jupyter nbconvert --to=python --output-dir='./' gene_symbol_sequences.ipynb", shell=True)
run(f"jupyter nbconvert --to=python --output-dir='./' ngram_analysis.ipynb", shell=True)
run(f"jupyter nbconvert --to=python --output-dir='./' classification.ipynb", shell=True)

# %%
# run(f"python gene_symbol_sequences.py", shell=True)
# run(f"python ngram_analysis.py", shell=True)
run(f"python classification.py", shell=True)

# %%



