# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on HIV

import torch.nn.functional as F

from ...model_zoo import *

__all__ = ['hiv_url',
           'create_hiv_model']

hiv_url = {
    'GCN_canonical_HIV': 'dgllife/pre_trained/gcn_canonical_hiv.pth',
    'GCN_attentivefp_HIV': 'dgllife/pre_trained/gcn_attentivefp_hiv.pth'
}

def create_hiv_model(model_name):
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
    if model_name == 'GCN_canonical_HIV':
        dropout = 0.0013086019242321
        return GCNPredictor(in_feats=74,
                            hidden_feats=[256],
                            activation=[F.relu],
                            residual=[False],
                            batchnorm=[True],
                            dropout=[dropout],
                            predictor_hidden_feats=512,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_HIV':
        num_gnn_layers = 3
        dropout = 0.010378057763216847
        return GCNPredictor(in_feats=39,
                            hidden_feats=[32] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[False] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=256,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    else:
        return None
