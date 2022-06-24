# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on BACE

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor, MPNNPredictor, \
    AttentiveFPPredictor, GINPredictor, NFPredictor

__all__ = ['bace_url',
           'create_bace_model']

bace_url = {
    'GCN_canonical_BACE': 'dgllife/pre_trained/gcn_canonical_bace.pth',
    'GCN_attentivefp_BACE': 'dgllife/pre_trained/gcn_attentivefp_bace.pth',
    'GAT_canonical_BACE': 'dgllife/pre_trained/gat_canonical_bace.pth',
    'GAT_attentivefp_BACE': 'dgllife/pre_trained/gat_attentivefp_bace_v2.pth',
    'Weave_canonical_BACE': 'dgllife/pre_trained/weave_canonical_bace.pth',
    'Weave_attentivefp_BACE': 'dgllife/pre_trained/weave_attentivefp_bace.pth',
    'MPNN_canonical_BACE': 'dgllife/pre_trained/mpnn_canonical_bace.pth',
    'MPNN_attentivefp_BACE': 'dgllife/pre_trained/mpnn_attentivefp_bace.pth',
    'AttentiveFP_canonical_BACE': 'dgllife/pre_trained/attentivefp_canonical_bace.pth',
    'AttentiveFP_attentivefp_BACE': 'dgllife/pre_trained/attentivefp_attentivefp_bace.pth',
    'gin_supervised_contextpred_BACE': 'dgllife/pre_trained/gin_supervised_contextpred_bace.pth',
    'gin_supervised_infomax_BACE': 'dgllife/pre_trained/gin_supervised_infomax_bace.pth',
    'gin_supervised_edgepred_BACE': 'dgllife/pre_trained/gin_supervised_edgepred_bace.pth',
    'gin_supervised_masking_BACE': 'dgllife/pre_trained/gin_supervised_masking_bace.pth',
    'NF_canonical_BACE': 'dgllife/pre_trained/nf_canonical_bace.pth'
}

def create_bace_model(model_name):
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

    if model_name == 'GCN_canonical_BACE':
        dropout = 0.022033656211803594
        return GCNPredictor(in_feats=74,
                            hidden_feats=[128],
                            activation=[F.relu],
                            residual=[True],
                            batchnorm=[False],
                            dropout=[dropout],
                            predictor_hidden_feats=16,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_BACE':
        dropout = 0.009923177126280991
        num_gnn_layers = 2
        return GCNPredictor(in_feats=39,
                            hidden_feats=[64] * num_gnn_layers,
                            activation=[F.relu] * num_gnn_layers,
                            residual=[False] * num_gnn_layers,
                            batchnorm=[False] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            predictor_hidden_feats=256,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_BACE':
        dropout = 0.012993892934328621
        return GATPredictor(in_feats=74,
                            hidden_feats=[64],
                            num_heads=[8],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.2547844032722401],
                            residuals=[False],
                            biases=[False],
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_BACE':
        dropout = 0.09842987062340869
        num_gnn_layers = 2
        return GATPredictor(in_feats=39,
                            hidden_feats=[256] * num_gnn_layers,
                            num_heads=[8] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.6702823790658061] * num_gnn_layers,
                            residuals=[False] * num_gnn_layers,
                            biases=[False] * num_gnn_layers,
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_BACE':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=2,
                              gnn_hidden_feats=32,
                              graph_feats=256,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_BACE':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=1,
                              gnn_hidden_feats=32,
                              graph_feats=32,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_BACE':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=64,
                             edge_hidden_feats=64,
                             num_step_message_passing=1,
                             num_step_set2set=3,
                             num_layer_set2set=1,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_BACE':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=64,
                             edge_hidden_feats=32,
                             num_step_message_passing=1,
                             num_step_set2set=1,
                             num_layer_set2set=1,
                             n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_canonical_BACE':
        return AttentiveFPPredictor(node_feat_size=74,
                                    edge_feat_size=13,
                                    num_layers=2,
                                    num_timesteps=4,
                                    graph_feat_size=16,
                                    dropout=0.39078446228187624,
                                    n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_attentivefp_BACE':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=11,
                                    num_layers=1,
                                    num_timesteps=4,
                                    graph_feat_size=32,
                                    dropout=0.12249297382460408,
                                    n_tasks=n_tasks)

    elif model_name == 'gin_supervised_contextpred_BACE':
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

    elif model_name == 'gin_supervised_infomax_BACE':
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

    elif model_name == 'gin_supervised_edgepred_BACE':
        jk = 'last'
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

    elif model_name == 'gin_supervised_masking_BACE':
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

    elif model_name == 'NF_canonical_BACE':
        num_gnn_layers = 1
        dropout = 0.14096514656248904
        return NFPredictor(in_feats=74,
                           n_tasks=n_tasks,
                           hidden_feats=[32] * num_gnn_layers,
                           batchnorm=[True] * num_gnn_layers,
                           dropout=[dropout] * num_gnn_layers,
                           predictor_hidden_size=1024,
                           predictor_batchnorm=True,
                           predictor_dropout=dropout)

    else:
        return None
