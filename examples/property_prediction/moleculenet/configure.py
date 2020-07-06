# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

# graph construction
from dgllife.utils import smiles_to_bigraph, smiles_to_complete_graph
# node featurization
from dgllife.utils import CanonicalAtomFeaturizer, WeaveAtomFeaturizer
# edge featurization
from dgllife.utils import WeaveEdgeFeaturizer

GCN_Tox21 = {
    'random_seed': 2,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'node_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gcn_hidden_feats': [64, 64],
    'predictor_hidden_feats': 64,
    'patience': 10,
    'smiles_to_graph': smiles_to_bigraph,
    'node_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc_score'
}

GAT_Tox21 = {
    'random_seed': 2,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'node_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gat_hidden_feats': [32, 32],
    'predictor_hidden_feats': 64,
    'num_heads': [4, 4],
    'patience': 10,
    'smiles_to_graph': smiles_to_bigraph,
    'node_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc_score'
}

Weave_Tox21 = {
    'random_seed': 2,
    'batch_size': 32,
    'lr': 1e-3,
    'num_epochs': 100,
    'node_data_field': 'h',
    'edge_data_field': 'e',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'num_gnn_layers': 2,
    'gnn_hidden_feats': 50,
    'graph_feats': 128,
    'patience': 10,
    'smiles_to_graph': partial(smiles_to_complete_graph, add_self_loop=True),
    'node_featurizer': WeaveAtomFeaturizer(),
    'edge_featurizer': WeaveEdgeFeaturizer(max_distance=2),
    'metric_name': 'roc_auc_score'
}

experiment_configures = {
    'GCN_Tox21': GCN_Tox21,
    'GAT_Tox21': GAT_Tox21,
    'Weave_Tox21': Weave_Tox21
}
def get_exp_configure(exp_name):
    return experiment_configures[exp_name]
