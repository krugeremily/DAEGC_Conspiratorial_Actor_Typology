# This code is based on TIGER101010's implementation of DAEGC in PyTorch 
# (https://github.com/Tiger101010/DAEGC/tree/main)

#################### DAEGC MODEL TO FINETUNE GAT NODE EMBEDDINGS BASED ON CLUSTERING ####################

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

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

from functions.daegc_helpers import create_adj_matrix,  create_feature_matrix, get_M, cluster_eval
from GAT import GAT

########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--samplesize', type=str, default='200', help = 'Total sample size combined from two datasets as int or "full"')
parser.add_argument('--max_epoch', type=int, default=50)
#parser.add_argument('--random_iter', type=int, default=10, help='Number of random search iterations')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_clusters', default=4, type=int)
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--embedding_size', default=16, type=int)
parser.add_argument('--weight_decay', type=int, default=5e-3)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--t_order', type = int, default = 2, help = 'Order of the transition matrix')
parser.add_argument('--loss_weight', type=float, default=10, help='Weight for KL Divergence loss')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
sample_size = args.samplesize

########## DAEGC MODEL ##########
class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrained model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer initialized with xavier
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    # get soft cluster assignment
    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q
    
    def forward(self, x, adj_norm, M):
        # get predicted adj matrix and node embeddings z
        A_pred, z = self.gat(x, adj_norm, M)
        # get soft cluster assignment
        q = self.get_Q(z)

        return A_pred, z, q

# compute target distribution from predicted cluster assignment for self-training
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

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
            writer.writerow([epoch, kl_loss, re_loss, loss.item(), sil_score, ch_score, db_score] + list(vars(args).values()))

            # Save model state
            torch.save(model.state_dict(), f'../../model/DAEGC_{date}/epoch_{epoch}.pkl')

if __name__ == "__main__":
    device = torch.device("cuda" if args.cuda else "cpu")

    dataset = pd.read_csv(f'../../data/samples/messages_sample_{sample_size}.csv.gzip', compression='gzip')
    agg_dataset = pd.read_csv(f'../../data/aggregated/author_{sample_size}.csv.gzip', compression='gzip')
    
    args.pretrain_path = '../../model/GAT_BASELINE_2024-10-11 13:00:23.519262/epoch_45.pkl'
    args.input_dim = len(agg_dataset.columns) - 3

    # initialize CSV file for saving performance metrics
    date = datetime.now()
    metrics_file = f'../../model/DAEGC_{date}/performance_metrics_{date}.csv'
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # write header
        writer.writerow(['Epoch', 'KL_Divergence','Recontruction Loss' ,'Ttotal Loss', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'] + list(vars(args).keys()))

        print(args)
        trainer(dataset, agg_dataset, args, writer)
