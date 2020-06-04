# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import torch

from dgl import DGLGraph

from dgllife.model.model_zoo import *

def test_graph3():
    """Graph with node features and edge features."""
    g = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g, torch.arange(g.number_of_nodes()).float().reshape(-1, 1), \
           torch.arange(2 * g.number_of_edges()).float().reshape(-1, 2)

def test_graph4():
    """Batched graph with node features and edge features."""
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    g2 = DGLGraph([(0, 1), (1, 2), (1, 3), (1, 4)])
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.number_of_nodes()).float().reshape(-1, 1), \
           torch.arange(2 * bg.number_of_edges()).float().reshape(-1, 2)

def test_attentivefp_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats, edge_feats = test_graph3()
    g, node_feats, edge_feats = g.to(device), node_feats.to(device), edge_feats.to(device)
    bg, batch_node_feats, batch_edge_feats = test_graph4()
    bg, batch_node_feats, batch_edge_feats = bg.to(device), batch_node_feats.to(device), \
                                             batch_edge_feats.to(device)
    attentivefp_predictor = AttentiveFPPredictor(node_feat_size=1,
                                                 edge_feat_size=2,
                                                 num_layers=2,
                                                 num_timesteps=1,
                                                 graph_feat_size=1,
                                                 n_tasks=2).to(device)
    assert attentivefp_predictor(g, node_feats, edge_feats).shape == torch.Size([1, 2])
    assert attentivefp_predictor(bg, batch_node_feats, batch_edge_feats).shape == \
           torch.Size([2, 2])
