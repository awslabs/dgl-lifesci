# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on HIV

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor, MPNNPredictor, \
    AttentiveFPPredictor, GINPredictor, NFPredictor

__all__ = ['hiv_url',
           'create_hiv_model']

hiv_url = {
    'GCN_canonical_HIV': 'dgllife/pre_trained/gcn_canonical_hiv.pth',
    'GCN_attentivefp_HIV': 'dgllife/pre_trained/gcn_attentivefp_hiv.pth',
    'GAT_canonical_HIV': 'dgllife/pre_trained/gat_canonical_hiv.pth',
    'GAT_attentivefp_HIV': 'dgllife/pre_trained/gat_attentivefp_hiv.pth',
    'Weave_canonical_HIV': 'dgllife/pre_trained/weave_canonical_hiv.pth',
    'Weave_attentivefp_HIV': 'dgllife/pre_trained/weave_attentivefp_hiv.pth',
    'MPNN_canonical_HIV': 'dgllife/pre_trained/mpnn_canonical_hiv_v2.pth',
    'MPNN_attentivefp_HIV': 'dgllife/pre_trained/mpnn_attentivefp_hiv_v2.pth',
    'AttentiveFP_canonical_HIV': 'dgllife/pre_trained/attentivefp_canonical_hiv.pth',
    'AttentiveFP_attentivefp_HIV': 'dgllife/pre_trained/attentivefp_attentivefp_hiv.pth',
    'gin_supervised_contextpred_HIV': 'dgllife/pre_trained/gin_supervised_contextpred_hiv.pth',
    'gin_supervised_infomax_HIV': 'dgllife/pre_trained/gin_supervised_infomax_hiv.pth',
    'gin_supervised_edgepred_HIV': 'dgllife/pre_trained/gin_supervised_edgepred_hiv.pth',
    'gin_supervised_masking_HIV': 'dgllife/pre_trained/gin_supervised_masking_hiv.pth',
    'NF_canonical_HIV': 'dgllife/pre_trained/nf_canonical_hiv.pth'
}

def create_hiv_model(model_name):
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
    if model_name == 'GCN_canonical_HIV':
        dropout = 0.0013086019242321
        return GCNPredictor(in_feats=74,
                            hidden_feats=[256],
                            activation=[F.relu],
                            residual=[False],
                            batchnorm=[True],
                            dropout=[dropout],
                            predictor_hidden_feats=512,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_HIV':
        num_gnn_layers = 3
        dropout = 0.010378057763216847
        return GCNPredictor(in_feats=39,
                            hidden_feats=[32] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[False] * num_gnn_layers,
                            batchnorm=[True] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=256,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_HIV':
        num_gnn_layers = 4
        dropout = 0.05181359502277236
        return GATPredictor(in_feats=74,
                            hidden_feats=[32] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.0821566804349384] * num_gnn_layers,
                            residuals=[False] * num_gnn_layers,
                            biases=[False] * num_gnn_layers,
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_HIV':
        num_gnn_layers = 3
        dropout = 0.06616256199038872
        return GATPredictor(in_feats=39,
                            hidden_feats=[256] * num_gnn_layers,
                            num_heads=[6] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.3649234413811788] * num_gnn_layers,
                            residuals=[False] * num_gnn_layers,
                            biases=[False] * num_gnn_layers,
                            predictor_hidden_feats=64,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_HIV':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=2,
                              gnn_hidden_feats=256,
                              graph_feats=64,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_HIV':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=4,
                              gnn_hidden_feats=64,
                              graph_feats=16,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_HIV':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=64,
                             edge_hidden_feats=64,
                             num_step_message_passing=4,
                             num_step_set2set=1,
                             num_layer_set2set=2,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_HIV':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=32,
                             edge_hidden_feats=64,
                             num_step_message_passing=3,
                             num_step_set2set=3,
                             num_layer_set2set=3,
                             n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_canonical_HIV':
        return AttentiveFPPredictor(node_feat_size=74,
                                    edge_feat_size=13,
                                    num_layers=1,
                                    num_timesteps=1,
                                    graph_feat_size=256,
                                    dropout=0.24511656823509329,
                                    n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_attentivefp_HIV':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=11,
                                    num_layers=1,
                                    num_timesteps=2,
                                    graph_feat_size=64,
                                    dropout=0.22938425755507835,
                                    n_tasks=n_tasks)

    elif model_name == 'gin_supervised_contextpred_HIV':
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

    elif model_name == 'gin_supervised_infomax_HIV':
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

    elif model_name == 'gin_supervised_edgepred_HIV':
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

    elif model_name == 'gin_supervised_masking_HIV':
        jk = 'sum'
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

    elif model_name == 'NF_canonical_HIV':
        dropout = 0.29642808718861385
        num_gnn_layers = 4
        return NFPredictor(in_feats=74,
                           n_tasks=n_tasks,
                           hidden_feats=[64] * num_gnn_layers,
                           batchnorm=[True] * num_gnn_layers,
                           dropout=[dropout] * num_gnn_layers,
                           predictor_hidden_size=256,
                           predictor_batchnorm=True,
                           predictor_dropout=dropout)

    else:
        return None
