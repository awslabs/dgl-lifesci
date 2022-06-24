# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained models on MUV

import torch.nn.functional as F

from ...model_zoo import GCNPredictor, GATPredictor, WeavePredictor, MPNNPredictor, \
    AttentiveFPPredictor, GINPredictor

__all__ = ['muv_url',
           'create_muv_model']

muv_url = {
    'GCN_canonical_MUV': 'dgllife/pre_trained/gcn_canonical_muv.pth',
    'GCN_attentivefp_MUV': 'dgllife/pre_trained/gcn_attentivefp_muv.pth',
    'GAT_canonical_MUV': 'dgllife/pre_trained/gat_canonical_muv.pth',
    'GAT_attentivefp_MUV': 'dgllife/pre_trained/gat_attentivefp_muv.pth',
    'Weave_canonical_MUV': 'dgllife/pre_trained/weave_canonical_muv.pth',
    'Weave_attentivefp_MUV': 'dgllife/pre_trained/weave_attentivefp_muv.pth',
    'MPNN_canonical_MUV': 'dgllife/pre_trained/mpnn_canonical_muv.pth',
    'MPNN_attentivefp_MUV': 'dgllife/pre_trained/mpnn_attentivefp_muv.pth',
    'AttentiveFP_canonical_MUV': 'dgllife/pre_trained/attentivefp_canonical_muv.pth',
    'AttentiveFP_attentivefp_MUV': 'dgllife/pre_trained/attentivefp_attentivefp_muv.pth',
    'gin_supervised_contextpred_MUV': 'dgllife/pre_trained/gin_supervised_contextpred_muv.pth',
    'gin_supervised_infomax_MUV': 'dgllife/pre_trained/gin_supervised_infomax_muv.pth',
    'gin_supervised_edgepred_MUV': 'dgllife/pre_trained/gin_supervised_edgepred_muv.pth',
    'gin_supervised_masking_MUV': 'dgllife/pre_trained/gin_supervised_masking_muv.pth'
}

def create_muv_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    n_tasks = 17
    if model_name == 'GCN_canonical_MUV':
        return GCNPredictor(in_feats=74,
                            hidden_feats=[32],
                            activation=[F.relu],
                            residual=[False],
                            batchnorm=[False],
                            dropout=[0.10811886971338101],
                            predictor_hidden_feats=128,
                            predictor_dropout=0.10811886971338101,
                            n_tasks=n_tasks)

    elif model_name == 'GCN_attentivefp_MUV':
        return GCNPredictor(in_feats=39,
                            hidden_feats=[64],
                            activation=[F.relu],
                            residual=[True],
                            batchnorm=[False],
                            dropout=[0.24997398695768708],
                            predictor_hidden_feats=128,
                            predictor_dropout=0.24997398695768708,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_canonical_MUV':
        num_gnn_layers = 4
        dropout = 0.5477918396466305
        return GATPredictor(in_feats=74,
                            hidden_feats=[128] * num_gnn_layers,
                            num_heads=[6] * num_gnn_layers,
                            feat_drops=[dropout] * num_gnn_layers,
                            attn_drops=[dropout] * num_gnn_layers,
                            alphas=[0.8145285541930105] * num_gnn_layers,
                            residuals=[True] * num_gnn_layers,
                            biases=[False] * num_gnn_layers,
                            predictor_hidden_feats=128,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'GAT_attentivefp_MUV':
        dropout = 0.37739180577199594
        return GATPredictor(in_feats=39,
                            hidden_feats=[128],
                            num_heads=[6],
                            feat_drops=[dropout],
                            attn_drops=[dropout],
                            alphas=[0.9101107032743763],
                            residuals=[False],
                            biases=[False],
                            predictor_hidden_feats=32,
                            predictor_dropout=dropout,
                            n_tasks=n_tasks)

    elif model_name == 'Weave_canonical_MUV':
        return WeavePredictor(node_in_feats=74,
                              edge_in_feats=13,
                              num_gnn_layers=1,
                              gnn_hidden_feats=64,
                              graph_feats=64,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'Weave_attentivefp_MUV':
        return WeavePredictor(node_in_feats=39,
                              edge_in_feats=11,
                              num_gnn_layers=3,
                              gnn_hidden_feats=32,
                              graph_feats=128,
                              gaussian_expand=False,
                              n_tasks=n_tasks)

    elif model_name == 'MPNN_canonical_MUV':
        return MPNNPredictor(node_in_feats=74,
                             edge_in_feats=13,
                             node_out_feats=64,
                             edge_hidden_feats=32,
                             num_step_message_passing=5,
                             num_step_set2set=2,
                             num_layer_set2set=3,
                             n_tasks=n_tasks)

    elif model_name == 'MPNN_attentivefp_MUV':
        return MPNNPredictor(node_in_feats=39,
                             edge_in_feats=11,
                             node_out_feats=32,
                             edge_hidden_feats=32,
                             num_step_message_passing=5,
                             num_step_set2set=2,
                             num_layer_set2set=1,
                             n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_canonical_MUV':
        return AttentiveFPPredictor(node_feat_size=74,
                                    edge_feat_size=13,
                                    num_layers=1,
                                    num_timesteps=3,
                                    graph_feat_size=16,
                                    dropout=0.20184515449053175,
                                    n_tasks=n_tasks)

    elif model_name == 'AttentiveFP_attentivefp_MUV':
        return AttentiveFPPredictor(node_feat_size=39,
                                    edge_feat_size=11,
                                    num_layers=1,
                                    num_timesteps=2,
                                    graph_feat_size=16,
                                    dropout=0.3260017176688692,
                                    n_tasks=n_tasks)

    elif model_name == 'gin_supervised_contextpred_MUV':
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

    elif model_name == 'gin_supervised_infomax_MUV':
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

    elif model_name == 'gin_supervised_edgepred_MUV':
        jk = 'max'
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

    elif model_name == 'gin_supervised_masking_MUV':
        jk = 'max'
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
