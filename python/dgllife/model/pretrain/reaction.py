# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models for reaction prediction

from ..model_zoo.wln_reaction_center import WLNReactionCenter
from ..model_zoo.wln_reaction_ranking import WLNReactionRanking

__all__ = ['reaction_url',
           'create_reaction_model']

reaction_url = {
    'wln_center_uspto': 'dgllife/pre_trained/wln_center_uspto_v3.pth',
    'wln_rank_uspto': 'dgllife/pre_trained/wln_rank_uspto.pth'
}

def create_reaction_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    if model_name == 'wln_center_uspto':
        return WLNReactionCenter(node_in_feats=82,
                                 edge_in_feats=6,
                                 node_pair_in_feats=10,
                                 node_out_feats=300,
                                 n_layers=3,
                                 n_tasks=5)

    elif model_name == 'wln_rank_uspto':
        return WLNReactionRanking(node_in_feats=89,
                                  edge_in_feats=5,
                                  node_hidden_feats=500,
                                  num_encode_gnn_layers=3)

    else:
        return None
