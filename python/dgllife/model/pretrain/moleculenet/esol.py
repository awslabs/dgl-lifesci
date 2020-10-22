# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on ESOL

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor, MPNNPredictor, \
    AttentiveFPPredictor, GINPredictor

__all__ = ['esol_url',
           'create_esol_model']

esol_url = {
    'GCN_canonical_ESOL': 'dgllife/pre_trained/gcn_canonical_esol.pth',
    'GCN_attentivefp_ESOL': 'dgllife/pre_trained/gcn_attentivefp_esol.pth'
}

def create_esol_model(model_name):
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

    if model_name == 'GCN_canonical_ESOL':
        dropout = 0.0004181672129021179
        return GCNPredictor(in_feats=74,
                            hidden_feats=[128],
                            activation=[F.relu],
                            residual=[True],
                            batchnorm=[False],
                            dropout=[dropout],
                            predictor_hidden_feats=1024,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_ESOL':
        dropout = 0.03400405080274294
        return GCNPredictor(in_feats=39,
                            hidden_feats=[64],
                            activation=[F.relu],
                            residual=[False],
                            batchnorm=[False],
                            dropout=[dropout],
                            predictor_hidden_feats=256,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    else:
        return None
