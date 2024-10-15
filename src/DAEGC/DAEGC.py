# This code is based on TIGER101010's implementation of DAEGC in PyTorch 
# (https://github.com/Tiger101010/DAEGC/tree/main)

#################### DAEGC MODEL TO FINETUNE GAT NODE EMBEDDINGS BASED ON CLUSTERING ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../functions')
sys.path.append('../../')

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


from GAT import GAT
from model_config import pretrain_path


########## DAEGC MODEL ##########
class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrained model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        self.gat.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

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
