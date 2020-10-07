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
    'GCN_canonical_FreeSolv': 'dgllife/pre_trained/gcn_canonical_freesolv_v2.pth'
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
        dropout = 0.006589580021982596
        return GCNPredictor(in_feats=74,
                            hidden_feats=[64],
                            activation=[F.relu],
                            residual=[False],
                            batchnorm=[False],
                            dropout=[dropout],
                            predictor_hidden_feats=1024,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    else:
        return None
