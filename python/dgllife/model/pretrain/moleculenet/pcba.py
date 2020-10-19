# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on PCBA

import torch.nn.functional as F

from ...model_zoo import *

__all__ = ['pcba_url',
           'create_pcba_model']

pcba_url = {
    'GCN_canonical_PCBA': 'dgllife/pre_trained/gcn_canonical_pcba.pth',
    'GCN_attentivefp_PCBA': 'dgllife/pre_trained/gcn_attentivefp_pcba.pth'
}

def create_pcba_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    n_tasks = 128

    if model_name == 'GCN_canonical_PCBA':
        num_gnn_layers = 2
        dropout = 0.053320999462421345
        return GCNPredictor(in_feats=74,
                            hidden_feats=[128] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=1024,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_PCBA':
        num_gnn_layers = 5
        dropout = 0.24826461148671453
        return GCNPredictor(in_feats=39,
                            hidden_feats=[128] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    else:
        return None
