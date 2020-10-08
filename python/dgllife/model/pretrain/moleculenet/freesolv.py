# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on FreeSolv

import torch.nn.functional as F

from ...model_zoo import *

__all__ = ['freesolv_url',
           'create_freesolv_model']

freesolv_url = {
    'GCN_canonical_FreeSolv': 'dgllife/pre_trained/gcn_canonical_freesolv_v2.pth',
    'GCN_attentivefp_FreeSolv': 'dgllife/pre_trained/gcn_attentivefp_freesolv.pth',
    'GAT_canonical_FreeSolv': 'dgllife/pre_trained/gat_canonical_freesolv.pth',
    'GAT_attentivefp_FreeSolv': 'dgllife/pre_trained/gat_attentivefp_freesolv.pth',
    'Weave_canonical_FreeSolv': 'dgllife/pre_trained/weave_canonical_freesolv.pth',
    'Weave_attentivefp_FreeSolv': 'dgllife/pre_trained/weave_attentivefp_freesolv.pth',
    'MPNN_canonical_FreeSolv': 'dgllife/pre_trained/mpnn_canonical_freesolv.pth',
    'MPNN_attentivefp_FreeSolv': 'dgllife/pre_trained/mpnn_attentivefp_freesolv.pth',
    'AttentiveFP_canonical_FreeSolv': 'dgllife/pre_trained/attentivefp_canonical_freesolv.pth',
    'AttentiveFP_attentivefp_FreeSolv':
        'dgllife/pre_trained/attentivefp_attentivefp_freesolv.pth',
}

def create_freesolv_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    n_tasks = 1

    if model_name == 'GCN_canonical_FreeSolv':
        dropout = 0.006589580021982596
        return GCNPredictor(in_feats=74,
                            hidden_feats=[64],
                            activation=[F.relu],
                            residual=[False],
                            batchnorm=[False],
                            dropout=[dropout],
                            predictor_hidden_feats=1024,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_FreeSolv':
        num_gnn_layers = 2
        dropout = 0.004282780783275264
        return GCNPredictor(in_feats=39,
                            hidden_feats=[32] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=256,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_FreeSolv':
        dropout = 0.13537547757851973
        return GATPredictor(in_feats=74,
                            hidden_feats=[256],
                            num_heads=[6],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.9199722462937526],
                            residuals=[False],
                            predictor_hidden_feats=16,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_FreeSolv':
        dropout = 0.3235550393975303
        num_gnn_layers = 4
        return GATPredictor(in_feats=39,
                            hidden_feats=[64] * num_gnn_layers,
                            num_heads=[4] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.8613751164365371] * num_gnn_layers,
                            residuals=[True] * num_gnn_layers,
                            predictor_hidden_feats=16,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_FreeSolv':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=2,
                              gnn_hidden_feats=32,
                              graph_feats=32,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_FreeSolv':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=2,
                              gnn_hidden_feats=128,
                              graph_feats=16,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_FreeSolv':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=32,
                             edge_hidden_feats=32,
                             num_step_message_passing=1,
                             num_step_set2set=2,
                             num_layer_set2set=2,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_FreeSolv':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=32,
                             edge_hidden_feats=32,
                             num_step_message_passing=3,
                             num_step_set2set=3,
                             num_layer_set2set=3,
                             n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_canonical_FreeSolv':
        return AttentiveFPPredictor(node_feat_size=74,
                                    edge_feat_size=13,
                                    num_layers=3,
                                    num_timesteps=3,
                                    graph_feat_size=128,
                                    dropout=0.4804978614147144,
                                    n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_attentivefp_FreeSolv':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=11,
                                    num_layers=3,
                                    num_timesteps=4,
                                    graph_feat_size=16,
                                    dropout=0.16687520888732352,
                                    n_tasks=n_tasks)

    else:
        return None
