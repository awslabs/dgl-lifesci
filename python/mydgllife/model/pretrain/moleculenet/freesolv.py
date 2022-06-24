# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on FreeSolv

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor, MPNNPredictor, \
    AttentiveFPPredictor, GINPredictor

__all__ = ['freesolv_url',
           'create_freesolv_model']

freesolv_url = {
    'GCN_canonical_FreeSolv': 'dgllife/pre_trained/gcn_canonical_freesolv_v3.pth',
    'GCN_attentivefp_FreeSolv': 'dgllife/pre_trained/gcn_attentivefp_freesolv_v2.pth',
    'GAT_canonical_FreeSolv': 'dgllife/pre_trained/gat_canonical_freesolv_v2.pth',
    'GAT_attentivefp_FreeSolv': 'dgllife/pre_trained/gat_attentivefp_freesolv_v2.pth',
    'Weave_canonical_FreeSolv': 'dgllife/pre_trained/weave_canonical_freesolv_v3.pth',
    'Weave_attentivefp_FreeSolv': 'dgllife/pre_trained/weave_attentivefp_freesolv_v2.pth',
    'MPNN_canonical_FreeSolv': 'dgllife/pre_trained/mpnn_canonical_freesolv_v2.pth',
    'MPNN_attentivefp_FreeSolv': 'dgllife/pre_trained/mpnn_attentivefp_freesolv_v2.pth',
    'AttentiveFP_canonical_FreeSolv': 'dgllife/pre_trained/attentivefp_canonical_freesolv_v2.pth',
    'AttentiveFP_attentivefp_FreeSolv':
        'dgllife/pre_trained/attentivefp_attentivefp_freesolv_v2.pth',
    'gin_supervised_contextpred_FreeSolv':
        'dgllife/pre_trained/gin_supervised_contextpred_freesolv_v2.pth',
    'gin_supervised_infomax_FreeSolv':
        'dgllife/pre_trained/gin_supervised_infomax_freesolv_v2.pth',
    'gin_supervised_edgepred_FreeSolv':
        'dgllife/pre_trained/gin_supervised_edgepred_freesolv_v2.pth',
    'gin_supervised_masking_FreeSolv':
        'dgllife/pre_trained/gin_supervised_masking_freesolv_v2.pth'
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
        num_gnn_layers = 2
        dropout = 0.05769700663189804
        return GCNPredictor(in_feats=74,
                            hidden_feats=[32] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_FreeSolv':
        num_gnn_layers = 4
        dropout = 0.09905316493862346
        return GCNPredictor(in_feats=39,
                            hidden_feats=[32] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=32,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_FreeSolv':
        dropout = 0.02327359604429937
        return GATPredictor(in_feats=74,
                            hidden_feats=[256],
                            num_heads=[4],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.6211392042947481],
                            residuals=[True],
                            biases=[False],
                            predictor_hidden_feats=256,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_FreeSolv':
        dropout = 0.06949846918000477
        num_gnn_layers = 2
        return GATPredictor(in_feats=39,
                            hidden_feats=[32] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.6294479518124414] * num_gnn_layers,
                            residuals=[True] * num_gnn_layers,
                            biases=[False] * num_gnn_layers,
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_FreeSolv':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=1,
                              gnn_hidden_feats=64,
                              graph_feats=64,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_FreeSolv':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=1,
                              gnn_hidden_feats=32,
                              graph_feats=16,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_FreeSolv':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=32,
                             edge_hidden_feats=32,
                             num_step_message_passing=4,
                             num_step_set2set=2,
                             num_layer_set2set=3,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_FreeSolv':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=32,
                             edge_hidden_feats=64,
                             num_step_message_passing=2,
                             num_step_set2set=2,
                             num_layer_set2set=1,
                             n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_canonical_FreeSolv':
        return AttentiveFPPredictor(node_feat_size=74,
                                    edge_feat_size=13,
                                    num_layers=4,
                                    num_timesteps=1,
                                    graph_feat_size=32,
                                    dropout=0.07118127568309571,
                                    n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_attentivefp_FreeSolv':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=11,
                                    num_layers=1,
                                    num_timesteps=1,
                                    graph_feat_size=128,
                                    dropout=0.1457037675069287,
                                    n_tasks=n_tasks)

    elif model_name == 'gin_supervised_contextpred_FreeSolv':
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

    elif model_name == 'gin_supervised_infomax_FreeSolv':
        jk = 'last'
        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=jk,
            dropout=0.5,
            readout='mean',
            n_tasks=n_tasks
        )
        model.gnn.JK = jk
        return model

    elif model_name == 'gin_supervised_edgepred_FreeSolv':
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

    elif model_name == 'gin_supervised_masking_FreeSolv':
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

    else:
        return None
