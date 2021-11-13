GCN_parallel = {
    'gnn_hidden_feats': 64,
    'num_gnn_layers': 2,
    'regressor_hidden_feats': 64,
    'lr': 1e-3,
    'weight_decay': 0.,
    'dropout': 0.,
    'patience': 50,
    'batch_size': 128
}

GCN_bypass = {
    'gnn_hidden_feats': 128,
    'num_gnn_layers': 2,
    'regressor_hidden_feats': 32,
    'lr': 1e-3,
    'weight_decay': 0.,
    'dropout': 0.,
    'patience': 30,
    'batch_size': 128
}

GAT_parallel = {
    'gnn_hidden_feats': 32,
    'num_gnn_layers': 2,
    'num_heads': 6,
    'regressor_hidden_feats': 32,
    'lr': 3e-3,
    'weight_decay': 3e-5,
    'dropout': 0.01,
    'patience': 100,
    'batch_size': 128
}

GAT_bypass = {
    'gnn_hidden_feats': 32,
    'num_gnn_layers': 3,
    'num_heads': 8,
    'regressor_hidden_feats': 32,
    'lr': 1e-3,
    'weight_decay': 3e-6,
    'dropout': 0.1,
    'patience': 30,
    'batch_size': 128
}

MPNN_parallel = {
    'node_hidden_dim': 64,
    'edge_hidden_dim': 16,
    'num_step_message_passing': 2,
    'num_step_set2set': 3,
    'num_layer_set2set': 2,
    'regressor_hidden_feats': 32,
    'lr': 1e-3,
    'weight_decay': 0.,
    'dropout': 0.,
    'patience': 50,
    'batch_size': 128
}

MPNN_bypass = {
    'node_hidden_dim': 32,
    'edge_hidden_dim': 64,
    'num_step_message_passing': 2,
    'num_step_set2set': 2,
    'num_layer_set2set': 2,
    'regressor_hidden_feats': 32,
    'lr': 1e-3,
    'weight_decay': 0.,
    'dropout': 0.01,
    'patience': 50,
    'batch_size': 128
}

AttentiveFP_parallel = {
    'num_gnn_layers': 3,
    'gnn_out_feats': 64,
    'num_timesteps': 3,
    'regressor_hidden_feats': 32,
    'lr': 1e-3,
    'weight_decay': 0.,
    'dropout': 0.,
    'patience': 50,
    'batch_size': 32
}

AttentiveFP_bypass = {
    'num_gnn_layers': 2,
    'gnn_out_feats': 32,
    'num_timesteps': 2,
    'regressor_hidden_feats': 32,
    'lr': 1e-3,
    'weight_decay': 0.,
    'dropout': 0.,
    'patience': 50,
    'batch_size': 32
}

configs = {
    'GCN_parallel': GCN_parallel,
    'GCN_bypass': GCN_bypass,
    'GAT_parallel': GAT_parallel,
    'GAT_bypass': GAT_bypass,
    'MPNN_parallel': MPNN_parallel,
    'MPNN_bypass': MPNN_bypass,
    'AttentiveFP_parallel': AttentiveFP_parallel,
    'AttentiveFP_bypass': AttentiveFP_bypass
}
