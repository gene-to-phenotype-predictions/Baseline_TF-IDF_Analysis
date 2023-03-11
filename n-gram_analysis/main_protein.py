# %%
import os as os
import sys as sys
import re as re
import pandas as pd
import numpy as np
import json as json
import pickle
import urllib as urllib
import zlib as zlib
import base64 as base64
from requests import Request, Session
import requests
from numpy import trapz
import itertools
import zlib as zlib
import base64 as base64
import pickle
import pathlib
from tqdm import tqdm
from datetime import datetime
import time
import copy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


pd.options.display.max_columns = 100
pd.options.display.min_rows = None
pd.options.display.max_rows = 10
pd.options.display.max_colwidth = 100

DATA_PATH = pathlib.Path('/data1/home/adpatter/gene-to-phenotype-predictions/adpatter/data/')
PROTEIN_SEQ_PATH = DATA_PATH.joinpath('gene_symbol_protein_sequences.pkl')

# %%
pd.read_pickle(PROTEIN_SEQ_PATH)

# %%
df = pd.read_pickle(PROTEIN_SEQ_PATH)

print(df.shape)

ds = df['seq'].drop_duplicates()

print(ds.shape)

corpus = ds.tolist()

ngram_count = {}
step = 1
n = 1
while True:

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n,n))

    X = vectorizer.fit_transform(corpus)

    count_vector = X.sum(axis=0)

    is_ones = (count_vector == 1)

    unique_count = is_ones.sum()

    print(n, X.shape[1], unique_count, abs(X.shape[1] - unique_count))

    ngram_count[n] = {'features': X.shape[1], 'unique_features': unique_count}

    with open(f'ngram_protein_{step}.pkl', 'wb') as f:
        pickle.dump(ngram_count, f, protocol=pickle.HIGHEST_PROTOCOL)

    if is_ones.all():
        break

    n = n + step


