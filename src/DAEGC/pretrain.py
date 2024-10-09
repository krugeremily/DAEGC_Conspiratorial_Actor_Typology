# This code is based on TIGER101010's implementation of DAEGC in PyTorch 
# (https://github.com/Tiger101010/DAEGC/tree/main)

#################### SCRIPT TO (PRE)TRAIN GAT MODEL FOR NODE EMBEDDINGS ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../functions')
sys.path.append('../../')

import argparse
import pandas as pd
from tqdm import tqdm
import csv
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterSampler

from functions.daegc_helpers import create_adj_matrix,  create_feature_matrix, get_M, cluster_eval
from GAT import GAT


########## SET PARAMETERS ##########

# default parameters
parser = argparse.ArgumentParser()
parser.add_argument('--samplesize', type=str, default='200', help = 'Total sample size combined from two datasets as int or "full"')
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--random_iter', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_clusters', default=4, type=int)
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--embedding_size', default=16, type=int)
parser.add_argument('--weight_decay', type=int, default=5e-3)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--t_order', type = int, default = 2, help = 'Order of the transition matrix')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
print(f'use cuda: {args.cuda}')
sample_size = args.samplesize

# for random search to find best parameters
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'n_clusters': [2, 4, 6],
    'hidden_size': [128, 256, 512],
    'embedding_size': [4, 8, 12],
    'weight_decay': [1e-4, 1e-3, 1e-2],
    'alpha': [0.1, 0.2, 0.3],
    't_order': [2, 4, 6]
}

########## PRETRAIN FUNCTION ##########

# load the dataset and pretrain the GAT model with Adam optimizer
def pretrain(dataset, agg_dataset, args, writer):
    # dataset is the not-aggregated data used to create the adjacency matrix; 
    # agg_dataset is the aggregated data used to create the feature matrix
    device = torch.device('cuda' if args.cuda else 'cpu')

    #initialize model and optimizer
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('Model initialized.')

    # process dataset to get adjacency matrix
    adj, adj_norm = create_adj_matrix(dataset)
    adj = adj.to(device)
    adj_norm = adj_norm.to(device)

    # get transition matrix
    M = get_M(adj_norm, t = args.t_order).to(device)

    # get feature matrix
    x = create_feature_matrix(agg_dataset)
    print('Adjacency Matrix, Transition Matrix and Feature matrix created.')

    # training loop
    for epoch in tqdm(range(args.max_epoch), desc='Training GAT'):
        model.train()
        A_pred, z = model(x, adj_norm, M)
        loss = F.mse_loss(A_pred.view(-1), adj.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clustering and evaluation
        with torch.no_grad():
            _, z = model(x, adj_norm, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )

        # save model sate & performance metrics every 5 epochs
        if epoch % 5 == 0:
            # to evaluate the clustering performance, get silhouette score, calinski harabasz score, and davies bouldin score
            sil_score, ch_score, db_score  = cluster_eval(x, kmeans.labels_)

            # save performance metrics
            writer.writerow([epoch, loss.item(), sil_score, ch_score, db_score] + list(vars(args).values()))

            # save model state
            torch.save(model.state_dict(), f'../../model/GAT_{date}/GAT_{epoch}.pkl')

########## MAIN FUNCTION ##########

if __name__ == "__main__":
    device = torch.device('cuda' if args.cuda else 'cpu')

    dataset = pd.read_csv(f'../../data/samples/messages_sample_{sample_size}.csv.gzip', compression='gzip')
    agg_dataset = pd.read_csv(f'../../data/aggregated/author_{sample_size}.csv.gzip', compression='gzip')

    args.input_dim = len(agg_dataset.columns) - 3 # subtract 3 for 'author', 'final_message_string', 'final_message'

    print(args)

    # Initialize CSV file for saving performance metrics
    date = datetime.now()
    metrics_file = f'../../model/GAT_{date}/performance_metrics_{date}.csv'
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Epoch', 'Loss', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'] + list(vars(args).keys()))

        # Perform random search
        for params in ParameterSampler(param_grid, n_iter=args.random_iter):
            for key, value in params.items():
                setattr(args, key, value)
            print(f'Training with parameters: {params}')
            pretrain(dataset, agg_dataset, args, writer)
