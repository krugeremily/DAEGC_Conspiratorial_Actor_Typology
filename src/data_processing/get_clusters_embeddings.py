#################### SCRIPT TO EXTRACT CLUSTER ASSIGNMENTS AND NODE EMBEDDINGS FROM DAEGC MODEL ####################

########## IMPORTS ##########
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../functions')
sys.path.append('../DAEGC')
sys.path.append('../../')
import argparse

from DAEGC import DAEGC
from functions.daegc_helpers import *
from model_config import model_path

########## SET PARAMETERS ##########
parser = argparse.ArgumentParser()
parser.add_argument('--samplesize', type=str, default='full', help = 'Total sample size combined from two datasets as int or "full"')
args = parser.parse_args()

sample_size = args.samplesize

########## LOAD AND PROCESS DATA ##########

data, agg_data = load_datasets(sample_size)
data['author'] = data['author'].astype(str)
agg_data['author'] = agg_data['author'].astype(str)

adj, adj_norm = create_adj_matrix(data)
x = create_feature_matrix(agg_data)
M = get_M(adj_norm)

print('Data, adjacency matrix, feature matrix, and M matrix loaded and processed.')

########## LOAD DAEGC MODEL ##########

# initialize DAEGC model
model = DAEGC(30, hidden_size=128, embedding_size=9, num_clusters=6)

# load pretrained model weights
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

print('DAEGC models loaded.')

########## GET CLUSTER ASSIGNMENTS AND NODE EMBEDDINGS ##########

# get embeddings & soft cluster assignments
with torch.no_grad():
    _, z, q = model(x, adj_norm, M)

labels = torch.argmax(q, dim=1)
cluster_count = Counter(labels.numpy())
print('Cluster Assignment for 4 Clusters:', cluster_count)

########## CONVERT ADJACENCY MATRIX TO EDGE LIST ##########

adj_numpy = adj.numpy()
# get weighted edge list
edge_list = pd.DataFrame()
edge_list['author_1'], edge_list['author_2'] = np.where(adj_numpy > 0)
edge_list['Weight'] = adj_numpy[np.where(adj_numpy > 0)]
edge_list.rename(columns={'author_1': 'source', 'author_2': 'target'}, inplace=True)

########## SAVE CLUSTER ASSIGNMENTS AND NODE EMBEDDINGS ##########

# remove text columns from aggregated dataset
agg_data = agg_data.drop(columns=['final_message', 'final_message_string'])

# save cluster assignments in aggregated dataset
agg_data['cluster'] = labels.numpy().tolist()

# save cluster probabilities in aggregated dataset
q_0 = q[:, 0]
q_1 = q[:, 1]
q_2 = q[:, 2]
q_3 = q[:, 3]
q_4 = q[:, 4]
q_5 = q[:, 5]

agg_data['cluster_0_prob'] = q_0.numpy().tolist()
agg_data['cluster_1_prob'] = q_1.numpy().tolist()
agg_data['cluster_2_prob'] = q_2.numpy().tolist()
agg_data['cluster_3_prob'] = q_3.numpy().tolist()
agg_data['cluster_4_prob'] = q_4.numpy().tolist()
agg_data['cluster_5_prob'] = q_5.numpy().tolist()

agg_data.to_csv(f'../../results/author_{sample_size}_features_and_clusters.csv', index=False)

# save only author and cluster columns for visualization in gephi
agg_data[['author', 'cluster']].to_csv(f'../../results/author_{sample_size}_clusters_only.csv', index=False)

# save node embeddings
np.save(f'../../results/author_{sample_size}_embeddings.npy', z.numpy())

# save adjacency matrix and edge list
edge_list.to_csv(f'../../results/author_{sample_size}_edge_list.csv.zip', compression= 'zip', index=False)
np.save(f'../../results/author_{sample_size}_adj.npy', adj.numpy())

print('Cluster assignments, node embeddings & Adjacency Matrix saved.')

