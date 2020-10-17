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
    'GCN_attentivefp_ToxCast': 'dgllife/pre_trained/gcn_attentivefp_toxcast.pth'
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

    else:
        return None
