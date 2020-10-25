# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on Lipophilicity

import torch.nn.functional as F

from ...model_zoo import GCNPredictor

__all__ = ['lipophilicity_url',
           'create_lipophilicity_model']

lipophilicity_url = {
    'GCN_canonical_Lipophilicity': 'dgllife/pre_trained/gcn_canonical_lipophilicity_v2.pth',
    'GCN_attentivefp_Lipophilicity': 'dgllife/pre_trained/gcn_attentivefp_lipophilicity_v2.pth',
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

    else:
        return None
