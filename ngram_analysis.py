#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os as os
import sys as sys
import re as re
import pandas as pd
import numpy as np
import json as json
import pickle
import pathlib
from datetime import datetime
import time
import copy
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns = 100
pd.options.display.min_rows = None
pd.options.display.max_rows = 10
pd.options.display.max_colwidth = 100

from config import RANDOM_STATE, MATERIALS_PATH, RESULTS_PATH

PROTEIN_SEQ_PATH = MATERIALS_PATH.joinpath('gene_symbol_protein_sequences.pkl')
EXON_SEQ_PATH = MATERIALS_PATH.joinpath('gene_symbol_dna_sequence_unspliced.pkl')
UNSPLICED_SEQ_PATH = MATERIALS_PATH.joinpath('gene_symbol_dna_sequence_exon.pkl')


# In[13]:


def ngram_procedure(ds, step, n, frac, name):

    ngram_count = {}

    ds = ds.sample(frac=frac)

    print(ds.shape)

    corpus = ds.tolist()

    while True:

        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n,n))

        X = vectorizer.fit_transform(corpus)

        count_vector = X.sum(axis=0)

        is_ones = (count_vector == 1)

        unique_count = is_ones.sum()

        print(n, X.shape[1], unique_count, abs(X.shape[1] - unique_count))

        ngram_count[n] = {'features': X.shape[1], 'unique_features': unique_count}

        with open(MATERIALS_PATH.joinpath(f'{name}_{step}_{frac}.pkl'), 'wb') as f:
            pickle.dump(ngram_count, f, protocol=pickle.HIGHEST_PROTOCOL)

        if is_ones.all():
            break

        n = n + step


# In[ ]:


df = pd.read_pickle(PROTEIN_SEQ_PATH)

ds = df['seq'].drop_duplicates()

ngram_procedure(ds=ds, step=1, n=1, frac=.01, 'ngram_protein')


# In[ ]:


df = pd.read_pickle(EXON_SEQ_PATH)

ds = df['Sequence'].drop_duplicates()

ngram_procedure(ds=ds, step=1, n=1, frac=.01, 'ngram_exon')


# In[ ]:


df = pd.read_pickle(UNSPLICED_SEQ_PATH)

ds = df['Sequence'].drop_duplicates()

ngram_procedure(ds=ds, step=1, n=1, frac=.01, 'ngram_unspliced')


# In[21]:


def transform_dict(path):

    with open(path, 'rb') as f:
        data = pickle.load(f)

        df = pd.DataFrame(data).T

        df = df.reset_index()

        df = df.rename({'index': 'ngram'}, axis=1)

        df['descriptive_features'] = df['features'] - df['unique_features']

        df = df.rename({'ngram': 'N-Gram', 'features': 'Features', 'unique_features': 'Unique Features', 'descriptive_features': 'Descriptive Features'}, axis=1)

        df = df.set_index(['N-Gram'])

    return df.copy()


# In[23]:


df_protein = transform_dict(MATERIALS_PATH.joinpath('ngram_protein_1_0.01.pkl'))
df_exon = transform_dict(MATERIALS_PATH.joinpath('ngram_exon_1_0.01.pkl'))
df_unspliced = transform_dict(MATERIALS_PATH.joinpath('ngram_unspliced_1_0.01.pkl'))


# In[24]:


sns.set_style('whitegrid')

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

fig.suptitle('N-Gram Analysis', fontsize='xx-large')
fig.set_size_inches((20,5))
fig.subplots_adjust(top=.85)

ax1.set_xlim(0, 10)
ax1.xaxis.set_ticks(range(0, 11, 1))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_title('Protein Features')

ax2.set_xlim(0, 20)
ax2.xaxis.set_ticks(range(0, 21, 2))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.set_title('Exon Features')

ax3.set_xlim(0, 20)
ax3.xaxis.set_ticks(range(0, 21, 2))
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.set_title('Unspliced Features')

_ = sns.lineplot(data=df_protein[['Features', 'Unique Features', 'Descriptive Features']], ax=ax1)
_= ax1.legend(loc='upper left')


_ = sns.lineplot(data=df_exon[['Features', 'Unique Features', 'Descriptive Features']], ax=ax2)
_ = ax2.legend(loc='upper left')

_ = sns.lineplot(data=df_unspliced[['Features', 'Unique Features', 'Descriptive Features']], ax=ax3)
_ = ax3.legend(loc='upper left')

plt.savefig(RESULTS_PATH.joinpath(f'ngram_analysis.png'), bbox_inches='tight')


# In[ ]:




