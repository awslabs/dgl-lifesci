# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on BBBP

import torch.nn.functional as F

from ...model_zoo import *

__all__ = ['bbbp_url',
           'create_bbbp_model']

bbbp_url = {
    'GCN_canonical_BBBP': 'dgllife/pre_trained/gcn_canonical_bbbp.pth',
    'GCN_attentivefp_BBBP': 'dgllife/pre_trained/gcn_attentivefp_bbbp.pth'
}

def create_bbbp_model(model_name):
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

    if model_name == 'GCN_canonical_BBBP':
        dropout = 0.0272564399565973
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

    elif model_name == 'GCN_attentivefp_BBBP':
        dropout = 0.2130511856011713
        num_gnn_layers = 2
        return GCNPredictor(in_feats=39,
                            hidden_feats=[128] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    else:
        return None
