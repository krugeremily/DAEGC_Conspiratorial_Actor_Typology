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

########## CALCULATE MEASURES ##########

# number of degrees
degrees = dict(G.degree())
agg_data['degree'] = agg_data['author'].map(degrees)
print('Degrees calculated.')

# degree centrality
degree_centrality = nx.degree_centrality(G)
agg_data['degree_centrality'] = agg_data['author'].map(degree_centrality)
print('Degree centrality calculated.')

# other centrality measures currently crash
# # closeness centrality
# closeness_centrality = nx.closeness_centrality(G)
# agg_data['closeness_centrality'] = agg_data['author'].map(closeness_centrality)
# print('Closeness centrality calculated.')

# # betweenness centrality
# betweenness_centrality = nx.betweenness_centrality(G)
# agg_data['betweenness_centrality'] = agg_data['author'].map(betweenness_centrality)
# print('Betweenness centrality calculated.')

# # eigenvector centrality
# eigenvector_centrality = nx.eigenvector_centrality(G)
# agg_data['eigenvector_centrality'] = agg_data['author'].map(eigenvector_centrality)
# print('Eigenvector centrality calculated.')

# save
agg_data.to_csv('../../results/author_full_features_and_clusters.csv.gzip', compression = 'gzip', index=False)