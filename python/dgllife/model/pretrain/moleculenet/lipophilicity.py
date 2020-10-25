# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on Lipophilicity

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor

__all__ = ['lipophilicity_url',
           'create_lipophilicity_model']

lipophilicity_url = {
    'GCN_canonical_Lipophilicity': 'dgllife/pre_trained/gcn_canonical_lipophilicity_v2.pth',
    'GCN_attentivefp_Lipophilicity': 'dgllife/pre_trained/gcn_attentivefp_lipophilicity_v2.pth',
    'GAT_canonical_Lipophilicity': 'dgllife/pre_trained/gat_canonical_lipophilicity.pth',
    'GAT_attentivefp_Lipophilicity': 'dgllife/pre_trained/gat_attentivefp_lipophilicity.pth'
}

def create_lipophilicity_model(model_name):
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

    if model_name == 'GCN_canonical_Lipophilicity':
        dropout = 0.28857669330071006
        num_gnn_layers = 2
        return GCNPredictor(in_feats=74,
                            hidden_feats=[128] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=16,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_Lipophilicity':
        dropout = 0.0690767663743611
        num_gnn_layers = 2
        return GCNPredictor(in_feats=39,
                            hidden_feats=[64] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_Lipophilicity':
        dropout = 0.00804563560011903
        num_gnn_layers = 5
        return GATPredictor(in_feats=74,
                            hidden_feats=[128] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.41300745504829595] * num_gnn_layers,
                            residuals=[False] * num_gnn_layers,
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_Lipophilicity':
        dropout = 0.023072031250834013
        return GATPredictor(in_feats=39,
                            hidden_feats=[128],
                            num_heads=[8],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.7133648170252214],
                            residuals=[False],
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    else:
        return None
