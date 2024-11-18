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
from model_config import model_path_3, model_path_4, model_path_5

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
model_3 = DAEGC(30, hidden_size=128, embedding_size=9, num_clusters=6)
model_4 = DAEGC(30, hidden_size=128, embedding_size=9, num_clusters=6)
model_5 = DAEGC(30, hidden_size=128, embedding_size=9, num_clusters=6)

# load pretrained model weights
model_3.load_state_dict(torch.load(model_path_3, map_location='cpu'))
model_3.eval()
model_4.load_state_dict(torch.load(model_path_4, map_location='cpu'))
model_4.eval()
model_5.load_state_dict(torch.load(model_path_5, map_location='cpu'))
model_5.eval()

print('DAEGC models loaded.')

########## GET CLUSTER ASSIGNMENTS AND NODE EMBEDDINGS ##########

# get embeddings & soft cluster assignments
with torch.no_grad():
    _, z_3, q_3 = model_3(x, adj_norm, M)
    _, z_4, q_4 = model_4(x, adj_norm, M)
    _, z_5, q_5 = model_5(x, adj_norm, M)

# get cluster assignments
q3_labels = torch.argmax(q_3, dim=1)
# count number of nodes in each cluster
q3_cluster_count = Counter(q3_labels.numpy())
print('Cluster Assignment for 3 Clusters:', q3_cluster_count)

q4_labels = torch.argmax(q_4, dim=1)
q4_cluster_count = Counter(q4_labels.numpy())
print('Cluster Assignment for 4 Clusters:', q4_cluster_count)

q5_labels = torch.argmax(q_5, dim=1)
q5_cluster_count = Counter(q5_labels.numpy())
print('Cluster Assignment for 5 Clusters:', q5_cluster_count)

########## CONVERT ADJACENCY MATRIX TO EDGE LIST ##########

adj_numpy = adj.numpy()
# get weighted edge list
edge_list = pd.DataFrame()
edge_list['author_1'], edge_list['author_2'] = np.where(adj_numpy > 0)
edge_list['Weight'] = adj_numpy[np.where(adj_numpy > 0)]
edge_list.rename(columns={'author_1': 'Source', 'author_2': 'Target'}, inplace=True)

########## SAVE CLUSTER ASSIGNMENTS AND NODE EMBEDDINGS ##########

# remove text columns from aggregated dataset
agg_data = agg_data.drop(columns=['final_message', 'final_message_string'])
# format labels as list
agg_data['cluster_3'] = q3_labels.numpy().tolist()
agg_data['cluster_4'] = q4_labels.numpy().tolist()
agg_data['cluster_5'] = q5_labels.numpy().tolist()

# save cluster assignments in aggregated dataset
agg_data['cluster_3'] = q3_labels
agg_data['cluster_4'] = q4_labels
agg_data['cluster_5'] = q5_labels

agg_data.to_csv(f'../../results/author_{sample_size}_features_and_clusters.csv', index=False)

# save only author and cluster columns for visualization in gephi
agg_data[['author', 'cluster_3', 'cluster_4', 'cluster_5']].to_csv(f'../../results/author_{sample_size}_clusters_only.csv', index=False)

# save node embeddings
np.save(f'../../results/author_{sample_size}_embeddings_3.npy', z_3.numpy())
np.save(f'../../results/author_{sample_size}_embeddings_4.npy', z_4.numpy())
np.save(f'../../results/author_{sample_size}_embeddings_5.npy', z_5.numpy())

# save adjacency matrix and edge list
edge_list.to_csv(f'../../results/author_{sample_size}_edge_list.csv.zip', compression= 'zip', index=False)
np.save(f'../../results/author_{sample_size}_adj.npy', adj.numpy())

print('Cluster assignments, node embeddings & Adjacency Matrix saved.')

