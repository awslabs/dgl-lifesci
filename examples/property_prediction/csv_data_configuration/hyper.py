# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hyperopt import hp

common_hyperparameters = {
    'lr': hp.uniform('lr', low=1e-4, high=3e-1),
    'weight_decay': hp.uniform('weight_decay', low=0, high=3e-3),
    'patience': hp.choice('patience', [30]),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512]),
}

gcn_hyperparameters = {
    'gnn_hidden_feats': hp.choice('gnn_hidden_feats', [32, 64, 128, 256]),
    'predictor_hidden_feats': hp.choice('predictor_hidden_feats', [16, 32, 64, 128, 256, 512, 1024]),
    'num_gnn_layers': hp.choice('num_gnn_layers', [1, 2, 3, 4, 5]),
    'residual': hp.choice('residual', [True, False]),
    'batchnorm': hp.choice('batchnorm', [True, False]),
    'dropout': hp.uniform('dropout', low=0., high=0.6)
}

gat_hyperparameters = {
    'gnn_hidden_feats': hp.choice('gnn_hidden_feats', [32, 64, 128, 256]),
    'num_heads': hp.choice('num_heads', [4, 6, 8]),
    'alpha': hp.uniform('alpha', low=0., high=1),
    'predictor_hidden_feats': hp.choice('predictor_hidden_feats', [16, 32, 64, 128, 256]),
    'num_gnn_layers': hp.choice('num_gnn_layers', [1, 2, 3, 4, 5]),
    'residual': hp.choice('residual', [True, False]),
    'dropout': hp.uniform('dropout', low=0., high=0.6)
}

weave_hyperparameters = {
    'gnn_hidden_feats': hp.choice('gnn_hidden_feats', [32, 64, 128, 256]),
    'num_gnn_layers': hp.choice('num_gnn_layers', [1, 2, 3, 4, 5]),
    'graph_feats': hp.choice('graph_feats', [16, 32, 64, 128, 256]),
    'gaussian_expand': hp.choice('gaussian_expand', [True, False]),
}

mpnn_hyperparameters = {
    'node_out_feats': hp.choice('node_out_feats', [32, 64]),
    'edge_hidden_feats': hp.choice('edge_hidden_feats', [32, 64]),
    'num_step_message_passing': hp.choice('num_step_message_passing', [1, 2, 3, 4, 5]),
    'num_step_set2set': hp.choice('num_step_set2set', [1, 2, 3]),
    'num_layer_set2set': hp.choice('num_layer_set2set', [1, 2, 3])
}

attentivefp_hyperparameters = {
    'num_layers': hp.choice('num_layers', [1, 2, 3, 4, 5]),
    'num_timesteps': hp.choice('num_timesteps', [1, 2, 3, 4, 5]),
    'graph_feat_size': hp.choice('graph_feat_size', [16, 32, 64, 128, 256]),
    'dropout': hp.uniform('dropout', low=0., high=0.6)
}

gin_pretrained_hyperparameters = {
    'jk': hp.choice('jk', ['concat', 'last', 'max', 'sum']),
    'readout': hp.choice('readout', ['sum', 'mean', 'max', 'attention'])
}

nf_hyperparameters = {
    'gnn_hidden_feats': hp.choice('gnn_hidden_feats', [32, 64, 128, 256]),
    'num_gnn_layers': hp.choice('num_gnn_layers', [1, 2, 3, 4, 5]),
    'batchnorm': hp.choice('batchnorm', [True, False]),
    'dropout': hp.uniform('dropout', low=0., high=0.6),
    'predictor_hidden_feats': hp.choice('predictor_hidden_feats', [16, 32, 64, 128, 256, 512, 1024])
}

def init_hyper_space(model):
    """Initialize the hyperparameter search space

    Parameters
    ----------
    model : str
        Model for searching hyperparameters

    Returns
    -------
    dict
        Mapping hyperparameter names to the associated search spaces
    """
    candidate_hypers = dict()
    candidate_hypers.update(common_hyperparameters)
    if model == 'GCN':
        candidate_hypers.update(gcn_hyperparameters)
    elif model == 'GAT':
        candidate_hypers.update(gat_hyperparameters)
    elif model == 'Weave':
        candidate_hypers.update(weave_hyperparameters)
    elif model == 'MPNN':
        candidate_hypers.update(mpnn_hyperparameters)
    elif model == 'AttentiveFP':
        candidate_hypers.update(attentivefp_hyperparameters)
    elif model in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                   'gin_supervised_edgepred', 'gin_supervised_masking']:
        candidate_hypers.update(gin_pretrained_hyperparameters)
    elif model == 'NF':
        candidate_hypers.update(nf_hyperparameters)
    else:
        return ValueError('Unexpected model: {}'.format(model))
    return candidate_hypers
