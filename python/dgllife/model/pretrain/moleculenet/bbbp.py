# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on BBBP

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor, MPNNPredictor, \
    AttentiveFPPredictor, GINPredictor, NFPredictor

__all__ = ['bbbp_url',
           'create_bbbp_model']

bbbp_url = {
    'GCN_canonical_BBBP': 'dgllife/pre_trained/gcn_canonical_bbbp.pth',
    'GCN_attentivefp_BBBP': 'dgllife/pre_trained/gcn_attentivefp_bbbp.pth',
    'GAT_canonical_BBBP': 'dgllife/pre_trained/gat_canonical_bbbp.pth',
    'GAT_attentivefp_BBBP': 'dgllife/pre_trained/gat_attentivefp_bbbp_v3.pth',
    'Weave_canonical_BBBP': 'dgllife/pre_trained/weave_canonical_bbbp.pth',
    'Weave_attentivefp_BBBP': 'dgllife/pre_trained/weave_attentivefp_bbbp_v2.pth',
    'MPNN_canonical_BBBP': 'dgllife/pre_trained/mpnn_canonical_bbbp.pth',
    'MPNN_attentivefp_BBBP': 'dgllife/pre_trained/mpnn_attentivefp_bbbp.pth',
    'AttentiveFP_canonical_BBBP': 'dgllife/pre_trained/attentivefp_canonical_bbbp.pth',
    'AttentiveFP_attentivefp_BBBP': 'dgllife/pre_trained/attentivefp_attentivefp_bbbp.pth',
    'gin_supervised_contextpred_BBBP': 'dgllife/pre_trained/gin_supervised_contextpred_bbbp.pth',
    'gin_supervised_infomax_BBBP': 'dgllife/pre_trained/gin_supervised_infomax_bbbp.pth',
    'gin_supervised_edgepred_BBBP': 'dgllife/pre_trained/gin_supervised_edgepred_bbbp.pth',
    'gin_supervised_masking_BBBP': 'dgllife/pre_trained/gin_supervised_masking_bbbp.pth',
    'NF_canonical_BBBP': 'dgllife/pre_trained/nf_canonical_bbbp.pth'
}

def create_bbbp_model(model_name):
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

    if model_name == 'GCN_canonical_BBBP':
        dropout = 0.0272564399565973
        num_gnn_layers = 4
        return GCNPredictor(in_feats=74,
                            hidden_feats=[256] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=32,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_BBBP':
        dropout = 0.2130511856011713
        num_gnn_layers = 2
        return GCNPredictor(in_feats=39,
                            hidden_feats=[128] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_BBBP':
        dropout = 0.046515821442611856
        num_gnn_layers = 3
        return GATPredictor(in_feats=74,
                            hidden_feats=[128] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.6544012585238377] * num_gnn_layers,
                            residuals=[False] * num_gnn_layers,
                            biases=[False] * num_gnn_layers,
                            predictor_hidden_feats=256,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_BBBP':
        dropout = 0.07001765207708285
        num_gnn_layers = 4
        return GATPredictor(in_feats=39,
                            hidden_feats=[32] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.8731920595699334] * num_gnn_layers,
                            residuals=[False] * num_gnn_layers,
                            biases=[False] * num_gnn_layers,
                            predictor_hidden_feats=256,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_BBBP':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=1,
                              gnn_hidden_feats=256,
                              graph_feats=256,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_BBBP':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=4,
                              gnn_hidden_feats=32,
                              graph_feats=256,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_BBBP':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=64,
                             edge_hidden_feats=64,
                             num_step_message_passing=1,
                             num_step_set2set=2,
                             num_layer_set2set=1,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_BBBP':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=64,
                             edge_hidden_feats=64,
                             num_step_message_passing=4,
                             num_step_set2set=3,
                             num_layer_set2set=2,
                             n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_canonical_BBBP':
        return AttentiveFPPredictor(node_feat_size=74,
                                    edge_feat_size=13,
                                    num_layers=1,
                                    num_timesteps=5,
                                    graph_feat_size=16,
                                    dropout=0.22184205119419326,
                                    n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_attentivefp_BBBP':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=11,
                                    num_layers=3,
                                    num_timesteps=2,
                                    graph_feat_size=128,
                                    dropout=0.4216675614776068,
                                    n_tasks=n_tasks)

    elif model_name == 'gin_supervised_contextpred_BBBP':
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

    elif model_name == 'gin_supervised_infomax_BBBP':
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

    elif model_name == 'gin_supervised_edgepred_BBBP':
        jk = 'sum'
        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=jk,
            dropout=0.5,
            readout='sum',
            n_tasks=n_tasks
        )
        model.gnn.JK = jk
        return model

    elif model_name == 'gin_supervised_masking_BBBP':
        jk = 'concat'
        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=jk,
            dropout=0.5,
            readout='sum',
            n_tasks=n_tasks
        )
        model.gnn.JK = jk
        return model

    elif model_name == 'NF_canonical_BBBP':
        num_gnn_layers = 2
        dropout = 0.1425900250956499
        return NFPredictor(in_feats=74,
                           n_tasks=n_tasks,
                           hidden_feats=[32] * num_gnn_layers,
                           batchnorm=[False] * num_gnn_layers,
                           dropout=[dropout] * num_gnn_layers,
                           predictor_hidden_size=32,
                           predictor_batchnorm=False,
                           predictor_dropout=dropout)

    else:
        return None
