################## FUNCTIONS USED IN TRAINING-LOOP FOR DAEGC ##################

########## IMPORTS ##########

import pandas as pd
import torch
import numpy as np
from itertools import combinations
from collections import Counter
from scipy.sparse import coo_matrix

########## CREATE ADJACENCY MATRIX FROM AGGREGATED DF ##########

def create_adj_matrix(dataset):
    ### CREATE EDGELIST BASED ON SHARED GROUP MEMBERSHIP ###
    grouped_authors = dataset.groupby('group_name')['author'].apply(set)

    # get unique authors and map them to indices
    authors = sorted(set(dataset['author']))
    author_idx_map = {author: idx for idx, author in enumerate(authors)}

    # get combinations of two authors in each group
    edges = []
    for authors_in_group in grouped_authors:
        if len(authors_in_group) > 1:
            edges += combinations(authors_in_group, 2)

    # count occurrences of each combination to determine edge weight
    edge_weights = Counter(edges)

    ### CREATE ADJACENCY MATRIX ###
    # Create empty lists for COO sparse matrix format (row, col, data)
    row_indices = []
    col_indices = []
    data = []

    for (author_1, author_2), weight in edge_weights.items():
        idx_1 = author_idx_map[author_1]
        idx_2 = author_idx_map[author_2]

        # Add both directions since the matrix is symmetric
        row_indices.append(idx_1)
        col_indices.append(idx_2)
        data.append(weight)
        
        row_indices.append(idx_2)
        col_indices.append(idx_1)
        data.append(weight)

    ########## CONVERT TO TENSOR ##########
    adj_tensor = torch.sparse_coo_tensor(
        indices=torch.tensor([row_indices, col_indices]),
        values=torch.tensor(data, dtype=torch.float32),
        size=(len(authors), len(authors))
    ).to_dense()

    print('Adjacency matrix created.')
    return adj_tensor

