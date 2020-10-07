# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on BACE

import torch.nn.functional as F

from ...model_zoo import *

__all__ = ['bace_url',
           'create_bace_model']

bace_url = {
    'GCN_canonical_BACE': 'dgllife/pre_trained/gcn_canonical_bace.pth',
}

def create_bace_model(model_name):
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

    if model_name == 'GCN_canonical_BACE':
        dropout = 0.022033656211803594
        return GCNPredictor(in_feats=74,
                            hidden_feats=[128],
                            activation=[F.relu],
                            residual=[True],
                            batchnorm=[False],
                            dropout=[dropout],
                            predictor_hidden_feats=16,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    else:
        return None
