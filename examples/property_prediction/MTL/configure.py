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
    'gnn_hidden_feats': 128,
    'num_gnn_layers': 3,
    'num_heads': 6,
    'regressor_hidden_feats': 32,
    'lr': 0.002,
    'weight_decay': 0.000013,
    'dropout': 0.069,
    'patience': 100,
    'batch_size': 256
}

GAT_bypass = {
    'gnn_hidden_feats': 64,
    'num_gnn_layers': 3,
    'num_heads': 8,
    'regressor_hidden_feats': 32,
    'lr': 0.00089,
    'weight_decay': 0.000039,
    'dropout': 0.039,
    'patience': 30,
    'batch_size': 256
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

GAT_LogD = {
    'gnn_hidden_feats': 16,
    'num_gnn_layers': 3,
    'num_heads': 8,
    'regressor_hidden_feats': 32,
    'lr': 0.0031,
    'weight_decay': 0.0000064,
    'dropout': 0.087,
    'patience': 100,
    'batch_size': 128
}

GAT_HLM = {
    'gnn_hidden_feats': 64,
    'num_gnn_layers': 2,
    'num_heads': 4,
    'regressor_hidden_feats': 16,
    'lr': 0.0032,
    'weight_decay': 0.000058,
    'dropout': 0.04,
    'patience': 100,
    'batch_size': 256
}

GAT_HH = {
    'gnn_hidden_feats': 64,
    'num_gnn_layers': 3,
    'num_heads': 6,
    'regressor_hidden_feats': 32,
    'lr': 0.00057,
    'weight_decay': 0.000042,
    'dropout': 0.063,
    'patience': 100,
    'batch_size': 256
}

GAT_KinSol = {
    'gnn_hidden_feats': 32,
    'num_gnn_layers': 3,
    'num_heads': 6,
    'regressor_hidden_feats': 16,
    'lr': 0.00038,
    'weight_decay': 0.000091,
    'dropout': 0.025,
    'patience': 100,
    'batch_size': 128
}

GCN_logD = {
    'gnn_hidden_feats': 128,
    'num_gnn_layers': 3,
    'regressor_hidden_feats': 128,
    'lr': 0.00015,
    'weight_decay': 0.00007,
    'dropout': 0.0859,
    'patience': 50,
    'batch_size': 256
}

GCN_HLM = {
    'gnn_hidden_feats': 64,
    'num_gnn_layers': 3,
    'regressor_hidden_feats': 64,
    'lr': 0.0013,
    'weight_decay': 0.000012,
    'dropout': 0.14,
    'patience': 50,
    'batch_size': 256
}

GCN_HH = {
    'gnn_hidden_feats': 128,
    'num_gnn_layers': 3,
    'regressor_hidden_feats': 32,
    'lr': 0.00855,
    'weight_decay': 0.00000086,
    'dropout': 0.059,
    'patience': 50,
    'batch_size': 128
}

GCN_KinSol = {
    'gnn_hidden_feats': 128,
    'num_gnn_layers': 3,
    'regressor_hidden_feats': 128,
    'lr': 0.0029,
    'weight_decay': 0.000017,
    'dropout': 0.035,
    'patience': 50,
    'batch_size': 256
}

MPNN_LogD = {
    'node_hidden_dim': 32,
    'edge_hidden_dim': 16,
    'num_step_message_passing': 3,
    'num_step_set2set': 3,
    'num_layer_set2set': 2,
    'regressor_hidden_feats': 32,
    'lr': 0.0064,
    'weight_decay': 0.00006,
    'dropout': 0.16,
    'patience': 50,
    'batch_size': 256
}

MPNN_HLM = {
    'node_hidden_dim': 128,
    'edge_hidden_dim': 64,
    'num_step_message_passing': 2,
    'num_step_set2set': 3,
    'num_layer_set2set': 2,
    'regressor_hidden_feats': 32,
    'lr': 0.0025,
    'weight_decay': 0.0000094,
    'dropout': 0.11,
    'patience': 50,
    'batch_size': 64
}

MPNN_HH = {
    'node_hidden_dim': 64,
    'edge_hidden_dim': 64,
    'num_step_message_passing': 3,
    'num_step_set2set': 2,
    'num_layer_set2set': 2,
    'regressor_hidden_feats': 64,
    'lr': 0.0069,
    'weight_decay': 0.00004,
    'dropout': 0.086,
    'patience': 50,
    'batch_size': 256
}

MPNN_KinSol = {
    'node_hidden_dim': 128,
    'edge_hidden_dim': 32,
    'num_step_message_passing': 3,
    'num_step_set2set': 2,
    'num_layer_set2set': 1,
    'regressor_hidden_feats': 32,
    'lr': 0.0036,
    'weight_decay': 0.000031,
    'dropout': 0.032,
    'patience': 50,
    'batch_size': 256
}

AttentiveFP_logD = {
    'num_gnn_layers': 3,
    'gnn_out_feats': 16,
    'num_timesteps': 3,
    'regressor_hidden_feats': 16,
    'lr': 0.0007,
    'weight_decay': 0.000078,
    'dropout': 0.086,
    'patience': 50,
    'batch_size': 64
}

AttentiveFP_HLM = {
    'num_gnn_layers': 2,
    'gnn_out_feats': 32,
    'num_timesteps': 3,
    'regressor_hidden_feats': 64,
    'lr': 0.0025,
    'weight_decay': 0.000032,
    'dropout': 0.067,
    'patience': 50,
    'batch_size': 256
}

AttentiveFP_HH = {
    'num_gnn_layers': 2,
    'gnn_out_feats': 64,
    'num_timesteps': 1,
    'regressor_hidden_feats': 16,
    'lr': 0.0029,
    'weight_decay': 0.000017,
    'dropout': 0.08,
    'patience': 50,
    'batch_size': 256
}

AttentiveFP_KinSol = {
    'num_gnn_layers': 3,
    'gnn_out_feats': 64,
    'num_timesteps': 1,
    'regressor_hidden_feats': 64,
    'lr': 0.002,
    'weight_decay': 0.000052,
    'dropout': 0.125,
    'patience': 50,
    'batch_size': 64
}
