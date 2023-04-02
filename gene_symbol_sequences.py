#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os as os
import sys as sys
import re as re
import pandas as pd
import numpy as np
import json as json
import pickle
import pathlib
from Bio import SeqIO

pd.options.display.max_columns = 100
pd.options.display.min_rows = None
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 100

from config import MATERIALS_PATH, RESULTS_PATH

FEATURE_PATH = MATERIALS_PATH.joinpath('mart_export.txt')

UNSPLICED_PATH = MATERIALS_PATH.joinpath('martquery_0219183013_823.txt')
UNSPLICED_RESULT_PATH = MATERIALS_PATH.joinpath('gene_symbol_dna_sequence_unspliced.pkl')

EXON_PATH = MATERIALS_PATH.joinpath('martquery_0225145730_334.txt')
EXON_RESULT_PATH = MATERIALS_PATH.joinpath('gene_symbol_dna_sequence_exon.pkl')

PROTEIN_FASTA_PATH = MATERIALS_PATH.joinpath('uniprot-compressed_true_download_true_format_fasta_includeIsoform_tr-2023.02.17-22.00.02.49.fasta')
PROTEIN_RESULT_PATH = MATERIALS_PATH.joinpath('gene_symbol_protein_sequences.pkl')


# In[2]:


df = pd.read_csv(FEATURE_PATH, sep='\t')

df = df[['Gene stable ID', 'Gene name']]

df = df.drop_duplicates()

_df_features = df.copy()

df


# In[3]:


records = []
for datum in SeqIO.parse(UNSPLICED_PATH, "fasta"):
    record = {}
    record['Gene stable ID'] = datum.description
    record['Sequence'] = str(datum.seq)
    records.append(record)

df = pd.DataFrame(records)

df = _df_features.merge(df, how='outer')

assert not df['Gene stable ID'].duplicated().any()

df.to_pickle(UNSPLICED_RESULT_PATH)

_df_unspliced = df.copy()


# In[4]:


_df_unspliced = pd.read_pickle(UNSPLICED_RESULT_PATH)

_df_unspliced


# In[5]:


_df_unspliced['Sequence'].str.len().max()


# In[6]:


records = []
for datum in SeqIO.parse(EXON_PATH, "fasta"):
    parts = datum.id.split('|')
    assert len(parts) == 2
    parts.append(str(datum.seq))
    records.append(parts)

df = pd.DataFrame(records, columns=['Gene stable ID', 'Exon stable ID', 'Sequence'])

df = _df_features.merge(df, how='outer')

assert not df['Exon stable ID'].duplicated().any()

df.to_pickle(EXON_RESULT_PATH)

_df_exon = df.copy()


# In[7]:


_df_exon = pd.read_pickle(EXON_RESULT_PATH)

_df_exon


# In[8]:


df = pd.read_csv(FEATURE_PATH, sep='\t')

df = df.groupby(['Gene Synonym']).filter(lambda x: x['Gene name'].dropna().drop_duplicates().shape[0] == 1)

df = df[['Gene Synonym', 'Gene name']]

df = df.drop_duplicates()

df = df.set_index(['Gene Synonym'])

ds = df['Gene name']

gene_synonym_gene_name = ds.to_dict()


# In[9]:


records = []
for data in SeqIO.parse(PROTEIN_FASTA_PATH, "fasta"):
    record={}
    for index, value in enumerate(re.findall(r'(?:(?<=^)[^|]+|(?<=\|)[^| ]+|(?<=\|)[^ ]+)|(?<=\ ).+?(?= \w\w\=|$)', data.description)):
        value = value.strip()
        if index == 0:
            record['db'] = value
        elif index == 1:
            record['UniqueIdentifier'] = value
        elif index == 2:
            record['EntryName'] = value
        elif index == 3:
            record['ProteinName'] = value
        else:
            k, v = re.split(r'\=', value)
            record[k] = v

    record['seq'] = str(data.seq)
    
    records.append(record)


# In[10]:


df = pd.DataFrame(records)

df['gene_symbol_harmonized'] = df['GN'].apply(lambda x: gene_synonym_gene_name.get(x, x))

df = df.set_index('gene_symbol_harmonized').reset_index()

_df_protein_sequence = df.copy()


# In[11]:


df = _df_protein_sequence.copy()

df = df[['gene_symbol_harmonized', 'db', 'UniqueIdentifier', 'EntryName', 'ProteinName', 'OS', 'OX', 'GN', 'PE', 'SV', 'seq']]

assert not df['UniqueIdentifier'].duplicated().any()

df.to_pickle(PROTEIN_RESULT_PATH)


# In[ ]:




