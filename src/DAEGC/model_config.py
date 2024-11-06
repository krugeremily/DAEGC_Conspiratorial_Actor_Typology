#################### FILE TO DEFINE RANDOM SEARCH VALUES ####################

# PARAMETER GRID FOR RANDOM SEARCH
param_grid_gat = {
	'lr': [0.001, 0.01, 0.0001],	
	'hidden_size': [64, 128, 256],	
	'embedding_size': [8, 9, 10, 11, 12, 13],
	'weight_decay': [1e-4, 1e-3, 1e-2]
}

# PARAMETER GRID FOR RANDOM SEARCH
param_grid_daegc = {
    'lr': [0.0001, 0.001, 0.01],
    'n_clusters': [4, 5, 6],
    'weight_decay': [1e-4, 1e-3, 5e-4, 5e-3],
    't_order': [2, 3, 4, 6]
}

pretrain_path = '../../model/GAT.pkl'