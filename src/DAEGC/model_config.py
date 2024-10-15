#################### FILE TO DEFINE RANDOM SEARCH VALUES ####################

# PARAMETER GRID FOR RANDOM SEARCH
param_grid_gat = {
    'lr': [0.001, 0.01, 0.1],
    'n_clusters': [2, 4, 6],
    'hidden_size': [128, 256, 512],
    'embedding_size': [4, 8, 12],
    'weight_decay': [1e-4, 1e-3, 1e-2],
    'alpha': [0.1, 0.2, 0.3],
    't_order': [2, 4, 6]
}

# PARAMETER GRID FOR RANDOM SEARCH
param_grid_daegc = {
    'lr': [0.0001, 0.001, 0.01, 0.1],
    'n_clusters': [3, 4, 5, 6],
    #'hidden_size': [128, 256, 512],
    #'embedding_size': [4, 8, 12],
    'weight_decay': [1e-4, 1e-3, 1e-2, 1e-1],
    'alpha': [0.05, 0.1, 0.2, 0.3],
    't_order': [2, 3, 4, 6]
}

pretrain_path = '../../model/GAT_BASELINE_2024-10-11 13:00:23.519262/epoch_45.pkl'