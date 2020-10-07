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
    'GCN_canonical_FreeSolv': 'dgllife/pre_trained/gcn_canonical_freesolv.pth'
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
        num_gnn_layers = 3
        dropout = 0.31707259912425456
        return GCNPredictor(in_feats=74,
                            hidden_feats=[256] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=1024,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    else:
        return None
