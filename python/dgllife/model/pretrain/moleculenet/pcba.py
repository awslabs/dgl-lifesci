# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on PCBA

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor, MPNNPredictor, \
    AttentiveFPPredictor, GINPredictor

__all__ = ['pcba_url',
           'create_pcba_model']

pcba_url = {
    'GCN_canonical_PCBA': 'dgllife/pre_trained/gcn_canonical_pcba.pth',
    'GCN_attentivefp_PCBA': 'dgllife/pre_trained/gcn_attentivefp_pcba.pth',
    'GAT_canonical_PCBA': 'dgllife/pre_trained/gat_canonical_pcba.pth',
    'GAT_attentivefp_PCBA': 'dgllife/pre_trained/gat_attentivefp_pcba.pth',
    'Weave_canonical_PCBA': 'dgllife/pre_trained/weave_canonical_pcba.pth',
    'Weave_attentivefp_PCBA': 'dgllife/pre_trained/weave_attentivefp_pcba.pth',
    'MPNN_canonical_PCBA': 'dgllife/pre_trained/mpnn_canonical_pcba.pth',
    'MPNN_attentivefp_PCBA': 'dgllife/pre_trained/mpnn_attentivefp_pcba.pth',
    'AttentiveFP_canonical_PCBA': 'dgllife/pre_trained/attentivefp_canonical_pcba.pth',
    'AttentiveFP_attentivefp_PCBA': 'dgllife/pre_trained/attentivefp_attentivefp_pcba.pth',
    'gin_supervised_contextpred_PCBA': 'dgllife/pre_trained/gin_supervised_contextpred_pcba.pth',
    'gin_supervised_infomax_PCBA': 'dgllife/pre_trained/gin_supervised_infomax_pcba.pth',
    'gin_supervised_edgepred_PCBA': 'dgllife/pre_trained/gin_supervised_edgepred_pcba.pth',
    'gin_supervised_masking_PCBA': 'dgllife/pre_trained/gin_supervised_masking_pcba.pth'
}

def create_pcba_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    n_tasks = 128

    if model_name == 'GCN_canonical_PCBA':
        num_gnn_layers = 2
        dropout = 0.053320999462421345
        return GCNPredictor(in_feats=74,
                            hidden_feats=[128] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=1024,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_PCBA':
        num_gnn_layers = 5
        dropout = 0.24826461148671453
        return GCNPredictor(in_feats=39,
                            hidden_feats=[128] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_PCBA':
        num_gnn_layers = 1
        dropout = 0.008451521225305653
        return GATPredictor(in_feats=74,
                            hidden_feats=[64] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.0194367227727808] * num_gnn_layers,
                            residuals=[False] * num_gnn_layers,
                            biases=[False] * num_gnn_layers,
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_PCBA':
        num_gnn_layers = 1
        dropout = 0.2811043226878611
        return GATPredictor(in_feats=39,
                            hidden_feats=[32] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.25837424873685433] * num_gnn_layers,
                            residuals=[True] * num_gnn_layers,
                            biases=[False] * num_gnn_layers,
                            predictor_hidden_feats=16,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_PCBA':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=1,
                              gnn_hidden_feats=128,
                              graph_feats=256,
                              gaussian_expand=True,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_PCBA':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=2,
                              gnn_hidden_feats=64,
                              graph_feats=64,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_PCBA':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=32,
                             edge_hidden_feats=64,
                             num_step_message_passing=1,
                             num_step_set2set=3,
                             num_layer_set2set=3,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_PCBA':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=64,
                             edge_hidden_feats=32,
                             num_step_message_passing=2,
                             num_step_set2set=1,
                             num_layer_set2set=1,
                             n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_canonical_PCBA':
        return AttentiveFPPredictor(node_feat_size=74,
                                    edge_feat_size=13,
                                    num_layers=2,
                                    num_timesteps=3,
                                    graph_feat_size=32,
                                    dropout=0.05370268638522968,
                                    n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_attentivefp_PCBA':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=11,
                                    num_layers=3,
                                    num_timesteps=2,
                                    graph_feat_size=16,
                                    dropout=0.31957324617702254,
                                    n_tasks=n_tasks)

    elif model_name == 'gin_supervised_contextpred_PCBA':
        jk = 'last'
        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=jk,
            dropout=0.5,
            readout='attention',
            n_tasks=n_tasks
        )
        model.gnn.JK = jk
        return model

    elif model_name == 'gin_supervised_infomax_PCBA':
        jk = 'concat'
        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=jk,
            dropout=0.5,
            readout='max',
            n_tasks=n_tasks
        )
        model.gnn.JK = jk
        return model

    elif model_name == 'gin_supervised_edgepred_PCBA':
        jk = 'sum'
        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=jk,
            dropout=0.5,
            readout='attention',
            n_tasks=n_tasks
        )
        model.gnn.JK = jk
        return model

    elif model_name == 'gin_supervised_masking_PCBA':
        jk = 'sum'
        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=jk,
            dropout=0.5,
            readout='attention',
            n_tasks=n_tasks
        )
        model.gnn.JK = jk
        return model

    else:
        return None
