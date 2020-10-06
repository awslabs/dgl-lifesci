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
    'GCN_attentivefp_MUV': 'dgllife/pre_trained/gcn_attentivefp_muv.pth'
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
    if model_name == 'GCN_canonical_MUV':
        return GCNPredictor(in_feats=74,
                            hidden_feats=[32],
                            activation=[F.relu],
                            residual=[False],
                            batchnorm=[False],
                            dropout=[0.10811886971338101],
                            predictor_hidden_feats=128,
                            predictor_dropout=0.10811886971338101,
                            n_tasks=17)

    elif model_name == 'GCN_attentivefp_MUV':
        return GCNPredictor(in_feats=39,
                            hidden_feats=[64],
                            activation=[F.relu],
                            residual=[True],
                            batchnorm=[False],
                            dropout=[0.24997398695768708],
                            predictor_hidden_feats=128,
                            predictor_dropout=0.24997398695768708,
                            n_tasks=17)

    else:
        return None
