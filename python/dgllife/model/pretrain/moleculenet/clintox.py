# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on ClinTox

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor, MPNNPredictor, \
    AttentiveFPPredictor

__all__ = ['clintox_url',
           'create_clintox_model']

clintox_url = {
    'GCN_canonical_ClinTox': 'dgllife/pre_trained/gcn_canonical_clintox.pth',
    'GCN_attentivefp_ClinTox': 'dgllife/pre_trained/gcn_attentivefp_clintox.pth',
    'GAT_canonical_ClinTox': 'dgllife/pre_trained/gat_canonical_clintox.pth',
    'GAT_attentivefp_ClinTox': 'dgllife/pre_trained/gat_attentivefp_clintox.pth',
    'Weave_canonical_ClinTox': 'dgllife/pre_trained/weave_canonical_clintox.pth',
    'Weave_attentivefp_ClinTox': 'dgllife/pre_trained/weave_attentivefp_clintox.pth',
    'MPNN_canonical_ClinTox': 'dgllife/pre_trained/mpnn_canonical_clintox.pth',
    'MPNN_attentivefp_ClinTox': 'dgllife/pre_trained/mpnn_attentivefp_clintox.pth',
    'AttentiveFP_canonical_ClinTox': 'dgllife/pre_trained/attentivefp_canonical_clintox.pth',
    'AttentiveFP_attentivefp_ClinTox': 'dgllife/pre_trained/attentivefp_attentivefp_clintox.pth'
}

def create_clintox_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    n_tasks = 2

    if model_name == 'GCN_canonical_ClinTox':
        dropout = 0.27771104411983266
        num_gnn_layers = 4
        return GCNPredictor(in_feats=74,
                            hidden_feats=[256] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=32,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_ClinTox':
        dropout = 0.09369442571380307
        num_gnn_layers = 5
        return GCNPredictor(in_feats=39,
                            hidden_feats=[32] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=512,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_ClinTox':
        dropout = 0.1622787886635157
        return GATPredictor(in_feats=74,
                            hidden_feats=[256],
                            num_heads=[4],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.4828530106865167],
                            residuals=[False],
                            biases=[False],
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_ClinTox':
        dropout = 0.023789159870020463
        return GATPredictor(in_feats=39,
                            hidden_feats=[64],
                            num_heads=[8],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.3794180901463749],
                            residuals=[True],
                            biases=[False],
                            predictor_hidden_feats=32,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_ClinTox':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=5,
                              gnn_hidden_feats=64,
                              graph_feats=32,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_ClinTox':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=5,
                              gnn_hidden_feats=64,
                              graph_feats=128,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_ClinTox':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=32,
                             edge_hidden_feats=32,
                             num_step_message_passing=4,
                             num_step_set2set=3,
                             num_layer_set2set=2,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_ClinTox':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=64,
                             edge_hidden_feats=32,
                             num_step_message_passing=2,
                             num_step_set2set=2,
                             num_layer_set2set=2,
                             n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_canonical_ClinTox':
        return AttentiveFPPredictor(node_feat_size=74,
                                    edge_feat_size=13,
                                    num_layers=2,
                                    num_timesteps=1,
                                    graph_feat_size=64,
                                    dropout=0.3391802249114625,
                                    n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_attentivefp_ClinTox':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=11,
                                    num_layers=1,
                                    num_timesteps=2,
                                    graph_feat_size=16,
                                    dropout=0.08746338896051695,
                                    n_tasks=n_tasks)

    else:
        return None
