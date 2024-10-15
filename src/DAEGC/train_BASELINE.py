# This code is based on TIGER101010's implementation of DAEGC in PyTorch 
# (https://github.com/Tiger101010/DAEGC/tree/main)

#################### SCRIPT TO TRAIN DAEGC MODEL (TO FINETUNE GAT NODE EMBEDDINGS BASED ON CLUSTERING) ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../functions')
sys.path.append('../../')

from tqdm import tqdm
import csv
from datetime import datetime

from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch.optim import Adam

from functions.daegc_helpers import parse_args, load_datasets, create_adj_matrix, create_feature_matrix, get_M, target_distribution, cluster_eval
from DAEGC import DAEGC
from model_config import pretrain_path

########## SET PARAMETERS ##########
args = parse_args()
args.cuda = torch.cuda.is_available()
print(f'use cuda: {args.cuda}')

########## TRAINING FUNCTION ##########

def trainer(dataset, agg_dataset, args, writer):
    # dataset is the not-aggregated data used to create the adjacency matrix; 
    # agg_dataset is the aggregated data used to create the feature matrix
    device = torch.device('cuda' if args.cuda else 'cpu')

    # initialize model and optimizer
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('DAEGC Model initialized.')

    # process dataset to get adjacency matrix
    adj, adj_norm = create_adj_matrix(dataset)
    adj = adj.to(device)
    adj_norm = adj_norm.to(device)

    # get transition matrix
    M = get_M(adj_norm, t = args.t_order).to(device)

    x = create_feature_matrix(agg_dataset)
    print('Adjacency Matrix, Transition Matrix and Feature matrix created.')

    # training loop
    for epoch in tqdm(range(args.max_epoch), desc='Training DAEGC'):
        model.train()
        # get predicted adj matrix, node embeddings z and soft cluster assignment Q
        A_pred, z, Q = model(x, adj_norm, M)
        q = Q.detach()  # Q

        p = target_distribution(Q.detach())

        # KL divergence loss for clustering
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        # reconstruction loss for node embeddings
        re_loss = F.mse_loss(A_pred.view(-1), adj.view(-1))

        # weighted sum of losses
        loss_weight = args.loss_weight
        loss = loss_weight * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save model state & performance metrics every 5 epochs
        if epoch % 5 == 0:
            kmeans = KMeans(n_clusters=args.n_clusters).fit(z.detach().cpu().numpy())
            sil_score, ch_score, db_score = cluster_eval(x, kmeans.labels_)

            # Save performance metrics
            writer.writerow([epoch, kl_loss.item(), re_loss.item(), loss.item(), sil_score, ch_score, db_score] + list(vars(args).values()))

            # Save model state
            torch.save(model.state_dict(), f'../../model/DAEGC_BASELINE_{date}/epoch_{epoch}.pkl')

if __name__ == '__main__':
    device = torch.device('cuda' if args.cuda else 'cpu')

    dataset, agg_dataset = load_datasets(args.samplesize)
    
    args.pretrain_path = pretrain_path
    args.input_dim = len(agg_dataset.columns) - 3

    # initialize CSV file for saving performance metrics
    date = datetime.now()
    metrics_file = f'../../model/DAEGC_BASELINE_{date}/performance_metrics_{date}.csv'
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # write header
        writer.writerow(['Epoch', 'KL_Divergence','Recontruction Loss' ,'Total Loss', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'] + list(vars(args).keys()))

        print(args)
        trainer(dataset, agg_dataset, args, writer)