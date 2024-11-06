# This code is based on TIGER101010's implementation of DAEGC in PyTorch 
# (https://github.com/Tiger101010/DAEGC/tree/main)

#################### GAT MODEL TO PRETRAIN NODE EMBEDDINGS ####################

########## IMPORTS ##########

import torch
import torch.nn as nn
import torch.nn.functional as F

########## GAT LAYER ##########
class GATLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # initialize weight and attention parameters for self and neighbors with xavier uniform (non-zero)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)


    def forward(self, input, adj, M, concat=True):
        
        h = torch.mm(input, self.W)

        # calculate attention for self and neighbors
        attn_for_self = torch.mm(h, self.a_self)  
        attn_for_neighs = torch.mm(h, self.a_neighs)  
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)  

        # mask out the zero values in the adjacency matrix
        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    # to print the layer
    def __repr__(self):
        return (
            self.__class__.__name__
            + ' ('
            + str(self.in_features)
            + ' -> '
            + str(self.out_features)
            + ')'
        )

########## GAT MODEL ##########

class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.conv1 = GATLayer(num_features, hidden_size)
        self.conv2 = GATLayer(hidden_size, embedding_size)

    # to compute dot product between embeddings to predict adjacency matrix
    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    # forward pass with two GAT layers
    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z