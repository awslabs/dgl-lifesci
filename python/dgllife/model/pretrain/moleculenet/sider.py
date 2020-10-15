# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on SIDER

import torch.nn.functional as F

from ...model_zoo import *

__all__ = ['sider_url',
           'create_sider_model']

sider_url = {
    'GCN_canonical_SIDER': 'dgllife/pre_trained/gcn_canonical_sider.pth',
    'GCN_attentivefp_SIDER': 'dgllife/pre_trained/gcn_attentivefp_sider.pth',
    'GAT_canonical_SIDER': 'dgllife/pre_trained/gat_canonical_sider.pth',
    'GAT_attentivefp_SIDER': 'dgllife/pre_trained/gat_attentivefp_sider.pth'
}

def create_sider_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    n_tasks = 27

    if model_name == 'GCN_canonical_SIDER':
        dropout = 0.034959769945995006
        num_gnn_layers = 3
        return GCNPredictor(in_feats=74,
                            hidden_feats=[256] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=512,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_SIDER':
        dropout = 0.08333992387843633
        num_gnn_layers = 4
        return GCNPredictor(in_feats=39,
                            hidden_feats=[256] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[False] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=1024,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_SIDER':
        dropout = 0.026789468731402546
        num_gnn_layers = 4
        return GATPredictor(in_feats=74,
                            hidden_feats=[256] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.7874749485670144] * num_gnn_layers,
                            residuals=[True] * num_gnn_layers,
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_SIDER':
        dropout = 0.5941912608794613
        num_gnn_layers = 4
        return GATPredictor(in_feats=39,
                            hidden_feats=[256] * num_gnn_layers,
                            num_heads=[4] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.15881060281037407] * num_gnn_layers,
                            residuals=[True] * num_gnn_layers,
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    else:
        return None
