#################### SCRIPT TO CREATE EDGELIST ####################

########## IMPORTS ##########
import time
import argparse
import os

import pandas as pd
from itertools import combinations
from collections import Counter

########## TIME ##########
start_time = time.time()

########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--samplesize', type=str, default='100', help = 'Total sample size combined from two datasets as int or "full"')
args = parser.parse_args()

sample_size = args.samplesize #sample size of loaded dataset

########## LOAD DATASET ##########
pre_agg = pd.read_csv(f'../../data/samples/messages_sample{sample_size}.csv.gzip', compression='gzip')
print('Dataset loaded.')

########## CREATE EDGELIST BASED ON SHARED GROUP MEMBERSHIP ##########
print('Creating edgelist.')
# get list of authors in each group
grouped_authors = pre_agg.groupby('group_name')['author'].apply(list)

# get combinations of two authors in each group
edges = []
for authors in grouped_authors:
    if len(authors) > 1:
        edges += combinations(sorted(set(authors)), 2)

# count occurences of combo to determine edge weight
edge_weights = Counter(edges)

# save as df
edgelist = pd.DataFrame(edge_weights.items(), columns=['edge', 'weight'])
edgelist[['author_1', 'author_2']] = pd.DataFrame(edgelist['edge'].tolist(), index=edgelist.index)
edgelist = edgelist.drop(columns='edge')
print('Edgelist created.')

########## CREATE ADJACENCY MATRIX ##########
# get list of unique authors
authors = sorted(set(pre_agg['author']))

# create adjacency matrix
adj_matrix = pd.DataFrame(0, index=authors, columns=authors)

# fill adjacency matrix with edge weights
for index, row in edgelist.iterrows():
    adj_matrix.loc[row['author_1'], row['author_2']] = row['weight']
    adj_matrix.loc[row['author_2'], row['author_1']] = row['weight']
print('Adjacency matrix created.')


os.create_dir('../../data/edgelists', exist_ok=True)
edgelist.to_csv(f'../../data/edgelists/author_{sample_size}_edgelist.csv', index=False)