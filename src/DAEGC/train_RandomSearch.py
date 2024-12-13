# This code is based on TIGER101010's implementation of DAEGC in PyTorch 
# (https://github.com/Tiger101010/DAEGC/tree/main)

#################### SCRIPT TO TRAIN DAEGC MODEL (TO FINETUNE GAT NODE EMBEDDINGS BASED ON CLUSTERING) ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../functions')
sys.path.append('../../')


import pandas as pd
from tqdm import tqdm
import csv
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterSampler

import torch
import torch.nn.functional as F
from torch.optim import Adam

from functions.daegc_helpers import parse_args, load_datasets, create_adj_matrix,  create_feature_matrix, get_M, target_distribution, cluster_eval
from DAEGC import DAEGC
from model_config import param_grid_daegc, pretrain_path

########## SET (DEFAULT) PARAMETERS ##########

args = parse_args()
args.cuda = torch.cuda.is_available()
print(f'use cuda: {args.cuda}')

def trainer(config):
    # set parameters
    adj = config['adj']
    adj_norm = config['adj_norm']
    M = config['M']
    x = config['x']
    args = config['args']
    writer = config['writer']
    date = config['date']
    iteration = config['iteration']
    state = config['state']

    device = torch.device('cuda' if args.cuda else 'cpu')

    # initialize model and optimizer
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, num_clusters=args.n_clusters).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('DAEGC Model initialized.')

    # training loop
    for epoch in tqdm(range(args.max_epoch), desc='Training DAEGC'):
        model.train()
        # get predicted adj matrix, node embeddings z and soft cluster assignment Q
        A_pred, z, Q = model(x, adj_norm, M)
        q = Q.detach()  # Q

        p = target_distribution(Q.detach())

        # reconstruction loss for node embeddings
        re_loss = F.mse_loss(A_pred.view(-1), adj.view(-1))
        # KL divergence loss for clustering
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

        # clustering evaluation
        q_labels = torch.argmax(q, dim=1).cpu().numpy()  
        # check for unique labels to prevent ValueError
        unique_labels = len(set(q_labels))
        if unique_labels > 1:
            # only calculate metrics if more than one unique label is present
            sil_score, ch_score, db_score = cluster_eval(z.detach().cpu().numpy(), q_labels)
        else:
            # if only one label, assign nan
            sil_score, ch_score, db_score = np.nan, np.nan, np.nan
            print(f'Skipped metrics at epoch {epoch + 1} due to single-class clustering.')

        if sil_score != np.nan:
            normalized_sil_score = (1 - sil_score) / 2
        else:
            normalized_sil_score = 10

        loss = kl_loss + re_loss + normalized_sil_score

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.writerow([epoch, kl_loss.item(), re_loss.item(), loss.item(), sil_score, ch_score, db_score, iteration, unique_labels] + list(vars(args).values()))

        # Save model state every 5 epochs and last epoch
        # if epoch % 5 == 0 or epoch == args.max_epoch - 1:
            # Save model state
            #torch.save(model.state_dict(), f'../../model/DAEGC_{args.state}_{date}/iter{iteration}_epoch_{epoch}.pkl')

if __name__ == '__main__':
    device = torch.device('cuda' if args.cuda else 'cpu')

    dataset, agg_dataset = load_datasets(args.samplesize)
    
    args.pretrain_path = pretrain_path
    args.input_dim = len(agg_dataset.columns) - 4 # subtract 4 for 'author', 'final_message_string', 'final_message', and 'avg_flesch_reading_ease_class'

    # initialize CSV file for saving performance metrics
    date = datetime.now()
    # change all non alphanumeric characters to underscore
    date = ''.join(e if e.isalnum() else '_' for e in str(date))
    metrics_file = f'../../model/DAEGC_{args.state}_{date}/performance_metrics_{date}.csv'
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)


    # process dataset to get adjacency matrix
    adj, adj_norm = create_adj_matrix(dataset)
    adj = adj.to(device)
    adj_norm = adj_norm.to(device)

    # get transition matrix
    M = get_M(adj_norm, t = args.t_order).to(device)

    x = create_feature_matrix(agg_dataset)
    print('Adjacency Matrix, Transition Matrix and Feature matrix created.')

    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # write header
        writer.writerow(['Epoch', 'KL_Divergence','Recontruction Loss' ,'Total Loss', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Random Searchh Iteration', 'Assigned Clusters'] + list(vars(args).keys()))

        # perform random search
        for iteration, params in tqdm(enumerate(ParameterSampler(param_grid_daegc, n_iter=args.random_iter)), desc='Random Search'):
            for key, value in params.items():
                setattr(args, key, value)
            print(f'Training with parameters: {params}')
            
            config = {
                'adj': adj,
                'adj_norm': adj_norm,
                'M': M,
                'x': x,
                'args': args,
                'writer': writer,
                'date': date,
                'iteration': iteration,
                'state': args.state,

            }

            trainer(config)