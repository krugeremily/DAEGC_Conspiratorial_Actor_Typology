################## FUNCTIONS USED IN TRAINING-LOOP FOR DAEGC ##################

########## IMPORTS ##########
import argparse
import pandas as pd
import torch
import numpy as np
from itertools import combinations
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

########## ARGUMENT PARSER ##########

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for node embeddings')
    # Add arguments
    parser.add_argument('--samplesize', type=str, default='200', help='Total sample size combined from two datasets as int or "full"')
    parser.add_argument('--max_epoch', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--random_iter', type=int, default=10, help='Number of random search iterations')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--n_clusters', type=int, default=4, help='Number of clusters for KMeans')
    parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in the GAT layers')
    parser.add_argument('--embedding_size', type=int, default=16, help='Size of the output embeddings')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay for regularization')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the leaky ReLU activation function')
    parser.add_argument('--t_order', type=int, default=2, help='Order of the transition matrix')
    parser.add_argument('--use_cuda', action='store_true', help='Flag to use CUDA if available')
    parser.add_argument('--loss_weight', type=float, default=10, help='Weight for KL Divergence loss')
    parser.add_argument('--layers', type=str, default='No additional layers', help='Additional Layers compared to baseline')
    
    return parser.parse_args()

########## LOAD DATASETS ##########

def load_datasets(sample_size):
    # Load the dataset and aggregated dataset
    dataset = pd.read_csv(f'../../data/samples/messages_sample_{sample_size}.csv.gzip', compression='gzip')
    agg_dataset = pd.read_csv(f'../../data/aggregated/author_{sample_size}.csv.gzip', compression='gzip')
    return dataset, agg_dataset

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
    # original adjacency matrix is used for performance eval
    adj_tensor = torch.sparse_coo_tensor(
        indices=torch.tensor([row_indices, col_indices]),
        values=torch.tensor(data, dtype=torch.float32),
        size=(len(authors), len(authors))
    ).to_dense()

    # normalized adjacency matrix with self-loop is used in the model
    adj_norm = adj_tensor + torch.eye(adj_tensor.shape[0])
    adj_norm = normalize(adj_norm.numpy(), norm='l1')
    adj_norm = torch.from_numpy(adj_norm).to(dtype=torch.float)

    return adj_tensor, adj_norm

########## CREATE FEATURE MATRIX FROM AGGREGATED DF ##########

def create_feature_matrix(dataset):
    dataset = dataset.fillna(0)
    # Create empty lists for COO sparse matrix format (row, col, data)
    row_indices = []
    col_indices = []
    data = []
    feature_columns = dataset.columns[1:]
    feature_columns = [feat for feat in feature_columns if (feat != 'final_message_string') & (feat != 'final_message')]

    # Create mapping of unique authors to indices
    authors = sorted(set(dataset['author']))
    author_idx_map = {author: idx for idx, author in enumerate(authors)}

    for idx, row in dataset.iterrows():
        for col, feature in enumerate(row[feature_columns]): 
            # to create a sparse matrix, only include non-zero features
            if feature != 0:
                row_indices.append(idx)
                col_indices.append(col)
                data.append(feature)

    ########## CONVERT TO TENSOR ##########
    feature_tensor = torch.sparse_coo_tensor(
        indices=torch.tensor([row_indices, col_indices]),
        values=torch.tensor(data, dtype=torch.float32),
        size=(len(dataset), len(feature_columns))
    ).to_dense()

    print('Feature matrix created.')
    return feature_tensor

########## GET TRANSITION MATRIX M ##########
def get_M(adj, t=2):
    adj_numpy = adj.cpu().numpy()
    tran_prob = normalize(adj_numpy, norm='l1', axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


########## GET TAREGT DISTRIBUTION FOR CLUSTERS ##########
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

########## EVALUATION FUNCTION ##########

def cluster_eval(data, clusters):
    sil_score = silhouette_score(data, clusters, metric='euclidean')
    ch_score = calinski_harabasz_score(data, clusters)
    db_score = davies_bouldin_score(data, clusters)
    return sil_score, ch_score, db_score