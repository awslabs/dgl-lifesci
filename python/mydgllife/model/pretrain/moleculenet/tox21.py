# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on Tox21

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor, MPNNPredictor, \
    AttentiveFPPredictor, GINPredictor, NFPredictor

__all__ = ['tox21_url',
           'create_tox21_model']

tox21_url = {
    'GCN_Tox21': 'dgllife/pre_trained/gcn_tox21.pth',
    'GAT_Tox21': 'dgllife/pre_trained/gat_tox21.pth',
    'Weave_Tox21': 'dgllife/pre_trained/weave_tox21.pth',
    'GCN_canonical_Tox21': 'dgllife/pre_trained/gcn_canonical_tox21.pth',
    'GCN_attentivefp_Tox21': 'dgllife/pre_trained/gcn_attentivefp_tox21.pth',
    'GAT_canonical_Tox21': 'dgllife/pre_trained/gat_canonical_tox21.pth',
    'GAT_attentivefp_Tox21': 'dgllife/pre_trained/gat_attentivefp_tox21.pth',
    'Weave_canonical_Tox21': 'dgllife/pre_trained/weave_canonical_tox21.pth',
    'Weave_attentivefp_Tox21': 'dgllife/pre_trained/weave_attentivefp_tox21.pth',
    'MPNN_canonical_Tox21': 'dgllife/pre_trained/mpnn_canonical_tox21.pth',
    'MPNN_attentivefp_Tox21': 'dgllife/pre_trained/mpnn_attentivefp_tox21.pth',
    'AttentiveFP_canonical_Tox21':
        'dgllife/pre_trained/attentivefp_canonical_tox21.pth',
    'AttentiveFP_attentivefp_Tox21':
        'dgllife/pre_trained/attentivefp_attentivefp_tox21.pth',
    'gin_supervised_contextpred_Tox21':
        'dgllife/pre_trained/gin_supervised_contextpred_tox21.pth',
    'gin_supervised_infomax_Tox21':
        'dgllife/pre_trained/gin_supervised_infomax_tox21.pth',
    'gin_supervised_edgepred_Tox21':
        'dgllife/pre_trained/gin_supervised_edgepred_tox21.pth',
    'gin_supervised_masking_Tox21':
        'dgllife/pre_trained/gin_supervised_masking_tox21.pth',
    'NF_canonical_Tox21': 'dgllife/pre_trained/nf_canonical_tox21.pth'
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
    n_tasks = 12

    if model_name == 'GCN_Tox21':
        return GCNPredictor(in_feats=74,
                            hidden_feats=[64, 64],
                            predictor_hidden_feats=64,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_Tox21':
        return GATPredictor(in_feats=74,
                            hidden_feats=[32, 32],
                            num_heads=[4, 4],
                            agg_modes=['flatten', 'mean'],
                            activations=[F.elu, None],
                            biases=[False, False],
                            predictor_hidden_feats=64,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_Tox21':
        return WeavePredictor(node_in_feats=27,
                              edge_in_feats=7,
                              num_gnn_layers=2,
                              gnn_hidden_feats=50,
                              graph_feats=128,
                              n_tasks=n_tasks)

    elif model_name == 'GCN_canonical_Tox21':
        dropout = 0.18118350615245202
        num_gnn_layers = 3
        return GCNPredictor(in_feats=74,
                            hidden_feats=[64] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[False] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=16,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_Tox21':
        dropout = 0.5432104441360837
        num_gnn_layers = 4
        return GCNPredictor(in_feats=39,
                            hidden_feats=[256] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[False] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=512,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_Tox21':
        dropout = 0.06205513003092991
        return GATPredictor(in_feats=74,
                            hidden_feats=[128],
                            num_heads=[4],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.574285650239047],
                            residuals=[True],
                            biases=[False],
                            predictor_hidden_feats=32,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_Tox21':
        dropout = 0.21647573234277548
        return GATPredictor(in_feats=39,
                            hidden_feats=[64],
                            num_heads=[4],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.3471639890634216],
                            residuals=[False],
                            biases=[False],
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_Tox21':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=5,
                              gnn_hidden_feats=256,
                              graph_feats=64,
                              gaussian_expand=True,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_Tox21':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=1,
                              gnn_hidden_feats=64,
                              graph_feats=256,
                              gaussian_expand=True,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_Tox21':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=32,
                             edge_hidden_feats=64,
                             num_step_message_passing=1,
                             num_step_set2set=3,
                             num_layer_set2set=3,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_Tox21':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=32,
                             edge_hidden_feats=64,
                             num_step_message_passing=3,
                             num_step_set2set=2,
                             num_layer_set2set=2,
                             n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_canonical_Tox21':
        return AttentiveFPPredictor(node_feat_size=74,
                                    edge_feat_size=13,
                                    num_layers=1,
                                    num_timesteps=5,
                                    graph_feat_size=32,
                                    dropout=0.02557007921295823,
                                    n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_attentivefp_Tox21':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=11,
                                    num_layers=2,
                                    num_timesteps=4,
                                    graph_feat_size=16,
                                    dropout=0.08321482571554469,
                                    n_tasks=n_tasks)

    elif model_name == 'gin_supervised_contextpred_Tox21':
        jk = 'concat'
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

    elif model_name == 'gin_supervised_infomax_Tox21':
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

    elif model_name == 'gin_supervised_edgepred_Tox21':
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

    elif model_name == 'gin_supervised_masking_Tox21':
        jk = 'concat'
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

    elif model_name == 'NF_canonical_Tox21':
        num_gnn_layers = 5
        dropout = 0.23946951437213781
        return NFPredictor(in_feats=74,
                           n_tasks=n_tasks,
                           hidden_feats=[256] * num_gnn_layers,
                           batchnorm=[True] * num_gnn_layers,
                           dropout=[dropout] * num_gnn_layers,
                           predictor_hidden_size=512,
                           predictor_batchnorm=True,
                           predictor_dropout=dropout)

    else:
        return None
