# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn.functional as F

from ...model_zoo import *

__all__ = ['muv_url',
           'create_muv_model']

muv_url = {
    'GCN_canonical_MUV': 'dgllife/pre_trained/gcn_canonical_muv.pth',
    'GCN_attentivefp_MUV': 'dgllife/pre_trained/gcn_attentivefp_muv.pth',
    'GAT_canonical_MUV': 'dgllife/pre_trained/gat_canonical_muv.pth',
    'GAT_attentivefp_MUV': 'dgllife/pre_trained/gat_attentivefp_muv.pth'
}

def create_muv_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    n_tasks = 17
    if model_name == 'GCN_canonical_MUV':
        return GCNPredictor(in_feats=74,
                            hidden_feats=[32],
                            activation=[F.relu],
                            residual=[False],
                            batchnorm=[False],
                            dropout=[0.10811886971338101],
                            predictor_hidden_feats=128,
                            predictor_dropout=0.10811886971338101,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_MUV':
        return GCNPredictor(in_feats=39,
                            hidden_feats=[64],
                            activation=[F.relu],
                            residual=[True],
                            batchnorm=[False],
                            dropout=[0.24997398695768708],
                            predictor_hidden_feats=128,
                            predictor_dropout=0.24997398695768708,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_MUV':
        num_gnn_layers = 4
        dropout = 0.5477918396466305
        return GATPredictor(in_feats=74,
                            hidden_feats=[128] * num_gnn_layers,
                            num_heads=[6] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.8145285541930105] * num_gnn_layers,
                            residuals=[True] * num_gnn_layers,
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_MUV':
        dropout = 0.37739180577199594
        return GATPredictor(in_feats=39,
                            hidden_feats=[128],
                            num_heads=[6],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.9101107032743763],
                            residuals=[False],
                            predictor_hidden_feats=32,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    else:
        return None
