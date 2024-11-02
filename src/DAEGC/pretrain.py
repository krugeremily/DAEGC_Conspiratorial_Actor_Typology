# This code is based on TIGER101010's implementation of DAEGC in PyTorch 
# (https://github.com/Tiger101010/DAEGC/tree/main)

#################### SCRIPT TO PRETRAIN GAT MODEL FOR NODE EMBEDDINGS ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../functions')
sys.path.append('../../')

from tqdm import tqdm
import csv
from datetime import datetime

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.cluster import KMeans

from functions.daegc_helpers import parse_args, load_datasets, create_adj_matrix,  create_feature_matrix, get_M, cluster_eval
from GAT import GAT


########## SET PARAMETERS ##########

args = parse_args()
args.cuda = torch.cuda.is_available()
print(f'use cuda: {args.cuda}')

########## PRETRAIN FUNCTION ##########

# load the dataset and pretrain the GAT model with Adam optimizer
def pretrain(dataset, agg_dataset, args=args):
    # dataset is the not-aggregated data used to create the adjacency matrix; 
    # agg_dataset is the aggregated data used to create the feature matrix

    #initialize model and optimizer
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=0.2,
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('Model initialized.')

    # process dataset to get adjacency matrix
    adj, adj_norm = create_adj_matrix(dataset)
    adj = adj.to(device)
    adj_norm = adj_norm.to(device)

    # get transition matrix
    M = get_M(adj_norm).to(device)

    # get feature matrix
    x = create_feature_matrix(agg_dataset)
    print('Adjacency Matrix, Transition Matrix and Feature matrix created.')

    # initialize CSV file for saving performance metrics
    date = datetime.now()
    # change all non alphanumeric characters to underscore
    date = ''.join(e if e.isalnum() else '_' for e in str(date))
    metrics_file = f'../../model/GAT_{args.state}_{date}/performance_metrics_{date}.csv'
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # write header
        writer.writerow(['Epoch', 'Loss', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'] + list(vars(args).keys()))


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
        
        # to evaluate the clustering performance, get silhouette score, calinski harabasz score, and davies bouldin score
        # check for unique labels to prevent ValueError
        unique_labels = len(set(kmeans.labels_))
        if unique_labels > 1:
            # only calculate metrics if more than one unique label is present
            sil_score, ch_score, db_score = cluster_eval(x, kmeans.labels_)
        else:
            # if only one label, assign nan
            sil_score, ch_score, db_score = np.nan, np.nan, np.nan
            print(f'Skipped metrics at epoch {epoch + 1} due to single-class clustering.')

        # save performance metrics
        with open(metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, loss.item(), sil_score, ch_score, db_score] + list(vars(args).values()))

        # save model sate every 5 epochs and last epoch
        if epoch % 5 == 0 or epoch == args.max_epoch - 1:
            torch.save(model.state_dict(), f'../../model/GAT_{args.state}_{date}/epoch_{epoch}.pkl')


########## MAIN FUNCTION ##########

if __name__ == '__main__':
    device = torch.device('cuda' if args.cuda else 'cpu')

    dataset, agg_dataset = load_datasets(args.samplesize)
    args.input_dim = len(agg_dataset.columns) - 4 # subtract 4 for 'author', 'final_message_string', 'final_message', and 'avg_flesch_reading_ease_class'

    print(args)
    pretrain(dataset, agg_dataset)
