# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on Tox21

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor

__all__ = ['tox21_url',
           'create_tox21_model']

tox21_url = {
    'GCN_Tox21': 'dgllife/pre_trained/gcn_tox21.pth',
    'GAT_Tox21': 'dgllife/pre_trained/gat_tox21.pth',
    'Weave_Tox21': 'dgllife/pre_trained/weave_tox21.pth'
}

def create_tox21_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    if model_name == 'GCN_Tox21':
        return GCNPredictor(in_feats=74,
                            hidden_feats=[64, 64],
                            predictor_hidden_feats=64,
                            n_tasks=12)

    elif model_name == 'GAT_Tox21':
        return GATPredictor(in_feats=74,
                            hidden_feats=[32, 32],
                            num_heads=[4, 4],
                            agg_modes=['flatten', 'mean'],
                            activations=[F.elu, None],
                            predictor_hidden_feats=64,
                            n_tasks=12)

    elif model_name == 'Weave_Tox21':
        return WeavePredictor(node_in_feats=27,
                              edge_in_feats=7,
                              num_gnn_layers=2,
                              gnn_hidden_feats=50,
                              graph_feats=128,
                              n_tasks=12)

    else:
        return None
