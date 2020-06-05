# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import torch
import torch.nn.functional as F

from dgl import DGLGraph

from dgllife.model.model_zoo import *

def test_graph1():
    """Graph with node features."""
    g = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g, torch.arange(g.number_of_nodes()).float().reshape(-1, 1)

def test_graph2():
    """Batched graph with node features."""
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    g2 = DGLGraph([(0, 1), (1, 2), (1, 3), (1, 4)])
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.number_of_nodes()).float().reshape(-1, 1)

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

def test_gat_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    # Test default setting
    gat_predictor = GATPredictor(in_feats=1).to(device)
    gat_predictor.eval()
    assert gat_predictor(g, node_feats).shape == torch.Size([1, 1])
    gat_predictor.train()
    assert gat_predictor(bg, batch_node_feats).shape == torch.Size([2, 1])

    # Test configured setting
    gat_predictor = GATPredictor(in_feats=1,
                                 hidden_feats=[1, 2],
                                 num_heads=[2, 3],
                                 feat_drops=[0.1, 0.1],
                                 attn_drops=[0.1, 0.1],
                                 alphas=[0.1, 0.1],
                                 residuals=[True, True],
                                 agg_modes=['mean', 'flatten'],
                                 activations=[None, F.elu]).to(device)
    gat_predictor.eval()
    assert gat_predictor(g, node_feats).shape == torch.Size([1, 1])
    gat_predictor.train()
    assert gat_predictor(bg, batch_node_feats).shape == torch.Size([2, 1])

if __name__ == '__main__':
    test_attentivefp_predictor()
    test_gat_predictor()
