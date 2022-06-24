# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on Lipophilicity

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor, MPNNPredictor, \
    AttentiveFPPredictor, GINPredictor

__all__ = ['lipophilicity_url',
           'create_lipophilicity_model']

lipophilicity_url = {
    'GCN_canonical_Lipophilicity': 'dgllife/pre_trained/gcn_canonical_lipophilicity_v2.pth',
    'GCN_attentivefp_Lipophilicity': 'dgllife/pre_trained/gcn_attentivefp_lipophilicity_v2.pth',
    'GAT_canonical_Lipophilicity': 'dgllife/pre_trained/gat_canonical_lipophilicity.pth',
    'GAT_attentivefp_Lipophilicity': 'dgllife/pre_trained/gat_attentivefp_lipophilicity.pth',
    'Weave_canonical_Lipophilicity': 'dgllife/pre_trained/weave_canonical_lipophilicity.pth',
    'Weave_attentivefp_Lipophilicity': 'dgllife/pre_trained/weave_attentivefp_lipophilicity.pth',
    'MPNN_canonical_Lipophilicity': 'dgllife/pre_trained/mpnn_canonical_lipophilicity.pth',
    'MPNN_attentivefp_Lipophilicity': 'dgllife/pre_trained/mpnn_attentivefp_lipophilicity.pth',
    'AttentiveFP_canonical_Lipophilicity':
        'dgllife/pre_trained/attentivefp_canonical_lipophilicity.pth',
    'AttentiveFP_attentivefp_Lipophilicity':
        'dgllife/pre_trained/attentivefp_attentivefp_lipophilicity.pth',
    'gin_supervised_contextpred_Lipophilicity':
        'dgllife/pre_trained/gin_supervised_contextpred_lipophilicity.pth',
    'gin_supervised_infomax_Lipophilicity':
        'dgllife/pre_trained/gin_supervised_infomax_lipophilicity.pth',
    'gin_supervised_edgepred_Lipophilicity':
        'dgllife/pre_trained/gin_supervised_edgepred_lipophilicity.pth',
    'gin_supervised_masking_Lipophilicity':
        'dgllife/pre_trained/gin_supervised_masking_lipophilicity.pth'
}

def create_lipophilicity_model(model_name):
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

    if model_name == 'GCN_canonical_Lipophilicity':
        dropout = 0.28857669330071006
        num_gnn_layers = 2
        return GCNPredictor(in_feats=74,
                            hidden_feats=[128] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=16,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_Lipophilicity':
        dropout = 0.0690767663743611
        num_gnn_layers = 2
        return GCNPredictor(in_feats=39,
                            hidden_feats=[64] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[True] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_Lipophilicity':
        dropout = 0.00804563560011903
        num_gnn_layers = 5
        return GATPredictor(in_feats=74,
                            hidden_feats=[128] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.41300745504829595] * num_gnn_layers,
                            residuals=[False] * num_gnn_layers,
                            biases=[False] * num_gnn_layers,
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_Lipophilicity':
        dropout = 0.023072031250834013
        return GATPredictor(in_feats=39,
                            hidden_feats=[128],
                            num_heads=[8],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.7133648170252214],
                            residuals=[False],
                            biases=[False],
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_Lipophilicity':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=2,
                              gnn_hidden_feats=64,
                              graph_feats=256,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_Lipophilicity':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=2,
                              gnn_hidden_feats=64,
                              graph_feats=128,
                              gaussian_expand=True,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_Lipophilicity':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=64,
                             edge_hidden_feats=32,
                             num_step_message_passing=2,
                             num_step_set2set=3,
                             num_layer_set2set=2,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_Lipophilicity':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=32,
                             edge_hidden_feats=32,
                             num_step_message_passing=3,
                             num_step_set2set=1,
                             num_layer_set2set=2,
                             n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_canonical_Lipophilicity':
        return AttentiveFPPredictor(node_feat_size=74,
                                    edge_feat_size=13,
                                    num_layers=2,
                                    num_timesteps=4,
                                    graph_feat_size=16,
                                    dropout=0.18905153162605368,
                                    n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_attentivefp_Lipophilicity':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=11,
                                    num_layers=1,
                                    num_timesteps=4,
                                    graph_feat_size=256,
                                    dropout=0.1392528529851128,
                                    n_tasks=n_tasks)

    elif model_name == 'gin_supervised_contextpred_Lipophilicity':
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

    elif model_name == 'gin_supervised_infomax_Lipophilicity':
        jk = 'max'
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

    elif model_name == 'gin_supervised_edgepred_Lipophilicity':
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

    elif model_name == 'gin_supervised_masking_Lipophilicity':
        jk = 'last'
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
