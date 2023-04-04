# %%
import os as os
import sys as sys
import re as re
import pandas as pd
import numpy as np
import json as json
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

pd.options.display.max_columns = 100
pd.options.display.min_rows = None
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 1000

from config import RANDOM_STATE, MATERIALS_PATH, RESULTS_PATH

GENE_SYMBOL_EFFECT_SIZE = MATERIALS_PATH.joinpath('capstone_body_weight_Statistical_effect_size_analysis_genotype_early_adult_scaled_13022023_gene_symbol_harmonized.pkl')
PROTEIN_SEQUENCE_PATH = MATERIALS_PATH.joinpath('gene_symbol_protein_sequences.pkl')
EXON_SEQUENCE_PATH = MATERIALS_PATH.joinpath('gene_symbol_dna_sequence_exon.pkl')
UNSPLICED_SEQUENCE_PATH = MATERIALS_PATH.joinpath('gene_symbol_dna_sequence_unspliced.pkl')

# %%
df = pd.read_pickle(GENE_SYMBOL_EFFECT_SIZE)

df = df.groupby(['gene_symbol_harmonized'])[['est_m_ea']].agg('mean')

df = df.reset_index()

df = df.sample(frac=.5)

kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE).fit(df[['est_m_ea']].to_numpy())

df['class'] = kmeans.labels_

assert not df['gene_symbol_harmonized'].duplicated().any()

_df_effect_size = df.copy()

print(df.shape)

# %%
df = pd.read_pickle(PROTEIN_SEQUENCE_PATH)

df = df.rename({'seq': 'sequence'}, axis=1)

df = df.groupby(['gene_symbol_harmonized'])[['sequence']].agg(lambda x: ' '.join(x.tolist()))

df = df.reset_index()

df = _df_effect_size.merge(df, how='inner')

df = df[['gene_symbol_harmonized', 'est_m_ea', 'class', 'sequence']]

print(df['class'].value_counts())

assert not df['gene_symbol_harmonized'].duplicated().any()

_df_protein = df.copy()

print(df.shape)

# %%
df = pd.read_pickle(EXON_SEQUENCE_PATH)

df = df.rename({'Sequence': 'sequence', 'Gene name': 'gene_symbol_harmonized'}, axis=1)

df = df.groupby(['gene_symbol_harmonized'])[['sequence']].agg(lambda x: ' '.join(x.tolist()))

df = df.reset_index()

df = _df_effect_size.merge(df, how='inner')

df = df[['gene_symbol_harmonized', 'est_m_ea', 'class', 'sequence']]

print(df['class'].value_counts())

assert not df['gene_symbol_harmonized'].duplicated().any()

_df_exon = df.copy()

print(df.shape)

# %%
df = pd.read_pickle(UNSPLICED_SEQUENCE_PATH)

df = df.rename({'Sequence': 'sequence', 'Gene name': 'gene_symbol_harmonized'}, axis=1)

df = df.groupby(['gene_symbol_harmonized'])[['sequence']].agg(lambda x: ' '.join(x.tolist()))

df = df.reset_index()

df = _df_effect_size.merge(df, how='inner')

df = df[['gene_symbol_harmonized', 'est_m_ea', 'class', 'sequence']]

print(df['class'].value_counts())

assert not df['gene_symbol_harmonized'].duplicated().any()

_df_unspliced = df.copy()

print(df.shape)

# %%
df = _df_protein.copy()

df = df.rename({'est_m_ea': 'Effect Size', 'class': 'Class'}, axis=1)

df['Protein'] = ''

_df_protein_strip_plot = df.copy()



df = _df_exon.copy()

df = df.rename({'est_m_ea': 'Effect Size', 'class': 'Class'}, axis=1)

df['Exon'] = ''

_df_exon_strip_plot = df.copy()



df = _df_unspliced.copy()

df = df.rename({'est_m_ea': 'Effect Size', 'class': 'Class'}, axis=1)

df['Unspliced'] = ''

_df_unspliced_strip_plot = df.copy()



sns.set_style('whitegrid')
fig, (ax1, ax2, ax3) = plt.subplots(3,1)

fig.suptitle('KMeans Clustering', fontsize='xx-large')
fig.set_size_inches((10,4.5))
fig.tight_layout()

_ = sns.stripplot(data=_df_protein_strip_plot, x='Effect Size', y='Protein', hue="Class", marker='.', jitter=True, s=2, ax=ax1, palette='tab10')
_= ax1.legend(loc='right')
_ = sns.stripplot(data=_df_exon_strip_plot, x='Effect Size', y='Exon', hue="Class", marker='.', jitter=True, s=2,  ax=ax2, palette='tab10')
_= ax2.legend(loc='right')
_ = sns.stripplot(data=_df_unspliced_strip_plot, x='Effect Size', y='Unspliced', hue="Class", marker='.', jitter=True, s=2,  ax=ax3, palette='tab10')
_= ax3.legend(loc='right')

plt.subplots_adjust(wspace=0.3, hspace=0.75)
plt.savefig(RESULTS_PATH.joinpath(f'kmeans_clustering_{RANDOM_STATE}.png'), bbox_inches='tight')

# %%
def feature_density(df, ngram_range):

    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range)

    X = tfidf.fit_transform(df['sequence'].tolist())

    X = X.todense()

    df = pd.DataFrame(X, columns=tfidf.get_feature_names_out())

    df = df.T.copy()

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE).fit(df.to_numpy())

    df['Class'] = kmeans.labels_

    print(pd.Series(kmeans.labels_).value_counts())

    df = df.groupby(['Class']).apply(lambda x: ((x != 0).sum().sum()/(x.shape[0] * x.shape[1]), x.index.tolist())).to_frame(name='density_features').reset_index()

    df['Density'] = df.apply(lambda x: x['density_features'][0], axis=1)

    df['Features'] = df.apply(lambda x: x['density_features'][1], axis=1)

    df['Count'] = df['Features'].apply(lambda x: len(x))

    df = df.drop(['density_features'], axis=1)

    return df.copy()

# %%
_df_protein_features = feature_density(df=_df_protein.copy(), ngram_range=(4,4))

# %%
_df_exon_features = feature_density(df=_df_exon.copy(), ngram_range=(10,10))

# %%
_df_unspliced_features = feature_density(df=_df_unspliced.sample(frac=.1, random_state=RANDOM_STATE).copy(), ngram_range=(12,12))

# %%
sns.set_style('white')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

fig.suptitle('Feature Density Analysis', fontsize='xx-large')
fig.set_size_inches((20,5))
fig.subplots_adjust(top=.85)

plt.subplots_adjust(wspace=0.3, hspace=0.3)

sns.barplot(data=_df_protein_features.copy(), x='Class', y='Count', alpha=0.5, ax=ax1, palette='tab10')
ax1.set_yscale("log")
ax1.set_title('Protein Feature Density')
ax1_twin = ax1.twinx()
sns.scatterplot(data=_df_protein_features, x='Class', y='Density', marker='d', label='Density', color = 'red', s=150, ax=ax1_twin, palette='tab10')
_= ax1_twin.legend(loc='upper right')

sns.barplot(data=_df_exon_features.copy(), x='Class', y='Count', alpha=0.5, ax=ax2, palette='tab10')
ax2.set_yscale("log")
ax2.set_title('Exon Feature Density')
ax2_twin = ax2.twinx()
sns.scatterplot(data=_df_exon_features, x='Class', y='Density', marker='d', label='Density', color = 'red', s=150, ax=ax2_twin, palette='tab10')
_= ax2_twin.legend(loc='upper right')

sns.barplot(data=_df_unspliced_features.copy(), x='Class', y='Count', alpha=0.5, ax=ax3, palette='tab10')
ax3.set_yscale("log")
ax3.set_title('Unspliced Feature Density')
ax3_twin = ax3.twinx()
sns.scatterplot(data=_df_unspliced_features, x='Class', y='Density', marker='d', label='Density', color = 'red', s=150, ax=ax3_twin, palette='tab10')
_= ax3_twin.legend(loc='upper right')

plt.savefig(RESULTS_PATH.joinpath(f'feature_density_analysis_{RANDOM_STATE}.png'), bbox_inches='tight')

# %%
def process(df, ngram_range, vocabulary):

    dfs = []

    X = df['sequence'].to_numpy()
    y = df['class'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)




    # KNeighborsClassifier
    param_grid = {
    'tfidfVectorizer__ngram_range': [ngram_range],
    'tfidfVectorizer__norm': ('l1', 'l2'),
    'kNeighborsClassifier__n_neighbors': [3, 5, 9]
    }

    pipe = Pipeline(steps=[
        ('tfidfVectorizer',  TfidfVectorizer(analyzer='char_wb', vocabulary=vocabulary)),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('kNeighborsClassifier', KNeighborsClassifier())
        ])

    kNeighborsClassifier = GridSearchCV(estimator=pipe, param_grid=param_grid, n_jobs=5, verbose=3)

    kNeighborsClassifier.fit(X_train, y_train)

    accuracy_score = kNeighborsClassifier.score(X_test, y_test)

    df = pd.DataFrame({
        'classifier': 'KNeighborsClassifier',
        'best_params_': [kNeighborsClassifier.best_params_],
        'best_score_': kNeighborsClassifier.best_score_,
        'test_score': accuracy_score
    })

    dfs.append(df)




    # SVC
    param_grid = {
        'tfidfVectorizer__ngram_range': [ngram_range],
        'tfidfVectorizer__norm': ('l1', 'l2'),
        'SVC__kernel': ['linear', 'rbf'],
        'SVC__C': [0.001,0.01,0.1,1,10,100, 1000],
        'SVC__gamma': ['scale', 'auto']
    }

    pipe = Pipeline(steps=[
        ('tfidfVectorizer',  TfidfVectorizer(analyzer='char_wb', vocabulary=vocabulary)),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('SVC', SVC())
        ])

    gsSVC = GridSearchCV(estimator=pipe, param_grid=param_grid, n_jobs=5, verbose=3)

    gsSVC.fit(X_train, y_train)

    accuracy_score = gsSVC.score(X_test, y_test)

    df = pd.DataFrame({
        'classifier': 'SVC',
        'best_params_': [gsSVC.best_params_],
        'best_score_': gsSVC.best_score_,
        'test_score': accuracy_score
    })
    
    dfs.append(df)




    # LogisticRegression
    param_grid = {
    'tfidfVectorizer__ngram_range': [ngram_range],
    'tfidfVectorizer__norm': ('l1', 'l2'),
    'logisticRegression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    pipe = Pipeline(steps=[
        ('tfidfVectorizer',  TfidfVectorizer(analyzer='char_wb', vocabulary=vocabulary)),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('logisticRegression', LogisticRegression())
        ])

    gsLogisticRegression = GridSearchCV(estimator=pipe, param_grid=param_grid, n_jobs=5, verbose=3)

    gsLogisticRegression.fit(X_train, y_train)

    accuracy_score = gsLogisticRegression.score(X_test, y_test)

    df = pd.DataFrame({
        'classifier': 'LogisticRegression',
        'best_params_': [gsLogisticRegression.best_params_],
        'best_score_': gsLogisticRegression.best_score_,
        'test_score': accuracy_score
    })

    dfs.append(df)




    # DummyClassifier
    param_grid = {
        'tfidfVectorizer__ngram_range': [ngram_range],
        'dummyClassifier__strategy': ['uniform', 'most_frequent']
    }

    pipe = Pipeline(steps=[
        ('tfidfVectorizer',  TfidfVectorizer(analyzer='char_wb', vocabulary=vocabulary)), 
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('dummyClassifier', DummyClassifier())
        ])

    gsDummyClassifier = GridSearchCV(estimator=pipe, param_grid=param_grid, n_jobs=5, verbose=3)

    gsDummyClassifier.fit(X_train, y_train)

    accuracy_score = gsDummyClassifier.score(X_test, y_test)

    df = pd.DataFrame({
        'classifier': 'DummyClassifier',
        'best_params_': [gsDummyClassifier.best_params_],
        'best_score_': gsDummyClassifier.best_score_,
        'test_score': accuracy_score
    })

    dfs.append(df)




    df = pd.concat(dfs)

    df = df.set_index(['classifier']).reset_index()

    return df.copy()

# %%
df = _df_protein_features.copy()

df = df.loc[df['Density'] != df['Density'].min()]

protein_vocabulary = df['Features'].sum()

df = _df_protein.copy()

df = process(df, (4,4), vocabulary=protein_vocabulary)

df.to_pickle(RESULTS_PATH.joinpath(f'classification_protein_{RANDOM_STATE}.pkl'))

# %%
df = _df_exon_features.copy()

df = df.loc[df['Density'] != df['Density'].min()]

exon_vocabulary = df['Features'].sum()

df = _df_exon.copy()

df = process(df, (10,10), vocabulary=exon_vocabulary)

df.to_pickle(RESULTS_PATH.joinpath(f'classification_exon_{RANDOM_STATE}.pkl'))

# %%
df = _df_unspliced_features.copy()

df = df.loc[df['Density'] != df['Density'].min()]

unspliced_vocabulary = df['Features'].sum()

df = _df_exon.copy()

df = _df_unspliced.copy()

df = process(df, (12,12), vocabulary=unspliced_vocabulary)

df.to_pickle(RESULTS_PATH.joinpath(f'classification_unspliced_{RANDOM_STATE}.pkl'))

