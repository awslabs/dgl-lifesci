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
    'GCN_attentivefp_ESOL': 'dgllife/pre_trained/gcn_attentivefp_esol.pth',
    'GAT_canonical_ESOL': 'dgllife/pre_trained/gat_canonical_esol.pth',
    'GAT_attentivefp_ESOL': 'dgllife/pre_trained/gat_attentivefp_esol.pth',
    'Weave_canonical_ESOL': 'dgllife/pre_trained/weave_canonical_esol.pth',
    'Weave_attentivefp_ESOL': 'dgllife/pre_trained/weave_attentivefp_esol.pth',
    'MPNN_canonical_ESOL': 'dgllife/pre_trained/mpnn_canonical_esol.pth',
    'MPNN_attentivefp_ESOL': 'dgllife/pre_trained/mpnn_attentivefp_esol.pth',
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

    elif model_name == 'GAT_canonical_ESOL':
        dropout = 0.28070328302954156
        return GATPredictor(in_feats=74,
                            hidden_feats=[32],
                            num_heads=[4],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.4994779445224584],
                            residuals=[True],
                            predictor_hidden_feats=16,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_ESOL':
        dropout = 0.00033036046538620356
        return GATPredictor(in_feats=39,
                            hidden_feats=[32],
                            num_heads=[8],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.7197105722372982],
                            residuals=[False],
                            predictor_hidden_feats=32,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_ESOL':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=3,
                              gnn_hidden_feats=256,
                              graph_feats=128,
                              gaussian_expand=True,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_ESOL':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=1,
                              gnn_hidden_feats=32,
                              graph_feats=256,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_ESOL':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=32,
                             edge_hidden_feats=64,
                             num_step_message_passing=3,
                             num_step_set2set=2,
                             num_layer_set2set=3,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_ESOL':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=32,
                             edge_hidden_feats=64,
                             num_step_message_passing=1,
                             num_step_set2set=2,
                             num_layer_set2set=2,
                             n_tasks=n_tasks)

    else:
        return None
