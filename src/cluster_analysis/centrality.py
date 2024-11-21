#################### SCRIPT TO CALCULATE CENTRALITY MEASURES FOR EACH ACTOR ####################

########## IMPORTS ##########
import networkx as nx
import pandas as pd
import numpy as np

########## LOAD DATA ##########

agg_data = pd.read_csv('../../results/author_full_features_and_clusters.csv')
adj = adj = np.load('../../results/author_full_adj.npy')
print('Data loaded.')

########## CREATE GRAPH ##########

G = nx.from_numpy_array(adj)
print('Graph created.')

########## CALCULATE CENTRALITY MEASURES ##########

# degree centrality
degree_centrality = nx.degree_centrality(G)
agg_data['degree_centrality'] = agg_data['author'].map(degree_centrality)
agg_data.to_csv('../../results/author_full_features_and_clusters.csv.gzip', compression = 'gzip', index=False)
print('Degree centrality calculated.')
