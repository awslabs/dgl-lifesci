# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on ToxCast

import torch.nn.functional as F

from ...model_zoo import *

__all__ = ['toxcast_url',
           'create_toxcast_model']

toxcast_url = {
    'GCN_canonical_ToxCast': 'dgllife/pre_trained/gcn_canonical_toxcast.pth',
    'GCN_attentivefp_ToxCast': 'dgllife/pre_trained/gcn_attentivefp_toxcast.pth',
    'GAT_canonical_ToxCast': 'dgllife/pre_trained/gat_canonical_toxcast.pth',
    'GAT_attentivefp_ToxCast': 'dgllife/pre_trained/gat_attentivefp_toxcast.pth',
    'Weave_canonical_ToxCast': 'dgllife/pre_trained/weave_canonical_toxcast.pth',
    'Weave_attentivefp_ToxCast': 'dgllife/pre_trained/weave_attentivefp_toxcast.pth',
    'MPNN_canonical_ToxCast': 'dgllife/pre_trained/mpnn_canonical_toxcast.pth',
    'MPNN_attentivefp_ToxCast': 'dgllife/pre_trained/mpnn_attentivefp_toxcast.pth'
}

def create_toxcast_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    n_tasks = 617
    if model_name == 'GCN_canonical_ToxCast':
        num_gnn_layers = 3
        dropout = 0.2354934715188964
        return GCNPredictor(in_feats=74,
                            hidden_feats=[256] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=512,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_ToxCast':
        dropout = 0.5790202973197223
        return GCNPredictor(in_feats=39,
                            hidden_feats=[256],
                            activation=[F.relu],
                            residual=[False],
                            batchnorm=[True],
                            dropout=[dropout],
                            predictor_hidden_feats=16,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_ToxCast':
        num_gnn_layers = 3
        dropout = 0.30477898651808644
        return GATPredictor(in_feats=74,
                            hidden_feats=[256] * num_gnn_layers,
                            num_heads=[6] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.5850073967467644] * num_gnn_layers,
                            residuals=[True] * num_gnn_layers,
                            predictor_hidden_feats=256,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_ToxCast':
        num_gnn_layers = 3
        dropout = 0.039304570924327153
        return GATPredictor(in_feats=39,
                            hidden_feats=[32] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.8044239663965763] * num_gnn_layers,
                            residuals=[False] * num_gnn_layers,
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_ToxCast':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=1,
                              gnn_hidden_feats=256,
                              graph_feats=256,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_ToxCast':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=1,
                              gnn_hidden_feats=128,
                              graph_feats=128,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_ToxCast':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=32,
                             edge_hidden_feats=32,
                             num_step_message_passing=2,
                             num_step_set2set=2,
                             num_layer_set2set=3,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_ToxCast':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=32,
                             edge_hidden_feats=64,
                             num_step_message_passing=5,
                             num_step_set2set=3,
                             num_layer_set2set=3,
                             n_tasks=n_tasks)

    else:
        return None
