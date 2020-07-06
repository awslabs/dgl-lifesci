# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

MPNN_Alchemy = {
    'random_seed': 0,
    'batch_size': 16,
    'node_in_feats': 15,
    'node_out_feats': 64,
    'edge_in_feats': 5,
    'edge_hidden_feats': 128,
    'n_tasks': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'mae',
    'mode': 'lower',
    'weight_decay': 0
}

SchNet_Alchemy = {
    'random_seed': 0,
    'batch_size': 16,
    'node_feats': 64,
    'hidden_feats': [64, 64, 64],
    'predictor_hidden_feats': 64,
    'n_tasks': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'mae',
    'mode': 'lower',
    'weight_decay': 0
}

MGCN_Alchemy = {
    'random_seed': 0,
    'batch_size': 16,
    'feats': 128,
    'n_layers': 3,
    'predictor_hidden_feats': 64,
    'n_tasks': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'mae',
    'mode': 'lower',
    'weight_decay': 0
}

experiment_configures = {
    'MPNN_Alchemy': MPNN_Alchemy,
    'SchNet_Alchemy': SchNet_Alchemy,
    'MGCN_Alchemy': MGCN_Alchemy
}
def get_exp_configure(exp_name):
    return experiment_configures[exp_name]
