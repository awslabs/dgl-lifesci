# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models for molecular property prediction

from ..gnn.gin import GIN
from ..model_zoo.attentivefp_predictor import AttentiveFPPredictor

__all__ = ['property_url',
           'create_property_model']

property_url = {
    'AttentiveFP_Aromaticity': 'dgllife/pre_trained/attentivefp_aromaticity.pth',
    'gin_supervised_contextpred': 'dgllife/pre_trained/gin_supervised_contextpred.pth',
    'gin_supervised_infomax': 'dgllife/pre_trained/gin_supervised_infomax.pth',
    'gin_supervised_edgepred': 'dgllife/pre_trained/gin_supervised_edgepred.pth',
    'gin_supervised_masking': 'dgllife/pre_trained/gin_supervised_masking.pth'
}

def create_property_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    if model_name == 'AttentiveFP_Aromaticity':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=10,
                                    num_layers=2,
                                    num_timesteps=2,
                                    graph_feat_size=200,
                                    n_tasks=1,
                                    dropout=0.2)

    elif model_name in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                        'gin_supervised_edgepred', 'gin_supervised_masking']:
        return GIN(num_node_emb_list=[120, 3],
                   num_edge_emb_list=[6, 3],
                   num_layers=5,
                   emb_dim=300,
                   JK='last',
                   dropout=0.5)

    else:
        return None
