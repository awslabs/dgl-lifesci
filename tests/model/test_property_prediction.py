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

def test_graph7():
    """Graph with categorical node and edge features."""
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g1, torch.LongTensor([0, 1, 0]), torch.LongTensor([2, 3, 4]), \
           torch.LongTensor([0, 0, 1]), torch.LongTensor([2, 3, 2])

def test_graph8():
    """Batched graph with categorical node and edge features."""
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    g2 = DGLGraph([(0, 1), (1, 2), (1, 3), (1, 4)])
    bg = dgl.batch([g1, g2])
    return bg, torch.LongTensor([0, 1, 0, 2, 1, 0, 2, 2]), \
           torch.LongTensor([2, 3, 4, 1, 0, 1, 2, 2]), \
           torch.LongTensor([0, 0, 1, 2, 1, 0, 0]), \
           torch.LongTensor([2, 3, 2, 0, 1, 2, 1])

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

def test_mlp_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g_feats = torch.tensor([[1.], [2.]]).to(device)
    mlp_predictor = MLPPredictor(in_feats=1, hidden_feats=1, n_tasks=2).to(device)
    assert mlp_predictor(g_feats).shape == torch.Size([2, 2])

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

def test_gcn_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    # Test default setting
    gcn_predictor = GCNPredictor(in_feats=1).to(device)
    gcn_predictor.eval()
    assert gcn_predictor(g, node_feats).shape == torch.Size([1, 1])
    gcn_predictor.train()
    assert gcn_predictor(bg, batch_node_feats).shape == torch.Size([2, 1])

    # Test configured setting
    gcn_predictor = GCNPredictor(in_feats=1,
                                 hidden_feats=[1],
                                 activation=[F.relu],
                                 residual=[True],
                                 batchnorm=[True],
                                 dropout=[0.1],
                                 classifier_hidden_feats=1,
                                 classifier_dropout=0.1,
                                 n_tasks=2).to(device)
    gcn_predictor.eval()
    assert gcn_predictor(g, node_feats).shape == torch.Size([1, 2])
    gcn_predictor.train()
    assert gcn_predictor(bg, batch_node_feats).shape == torch.Size([2, 2])

def test_gin_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats1, node_feats2, edge_feats1, edge_feats2 = test_graph7()
    node_feats1, node_feats2 = node_feats1.to(device), node_feats2.to(device)
    edge_feats1, edge_feats2 = edge_feats1.to(device), edge_feats2.to(device)
    bg, batch_node_feats1, batch_node_feats2, \
    batch_edge_feats1, batch_edge_feats2 = test_graph8()
    batch_node_feats1, batch_node_feats2 = batch_node_feats1.to(device), \
                                           batch_node_feats2.to(device)
    batch_edge_feats1, batch_edge_feats2 = batch_edge_feats1.to(device), \
                                           batch_edge_feats2.to(device)

    num_node_emb_list = [3, 5]
    num_edge_emb_list = [3, 4]
    for JK in ['concat', 'last', 'max', 'sum']:
        for readout in ['sum', 'mean', 'max', 'attention']:
            model = GINPredictor(num_node_emb_list=num_node_emb_list,
                                 num_edge_emb_list=num_edge_emb_list,
                                 num_layers=2,
                                 emb_dim=10,
                                 JK=JK,
                                 readout=readout,
                                 n_tasks=2).to(device)
            assert model(g, [node_feats1, node_feats2], [edge_feats1, edge_feats2]).shape \
                   == torch.Size([1, 2])
            assert model(bg, [batch_node_feats1, batch_node_feats2],
                         [batch_edge_feats1, batch_edge_feats2]).shape == torch.Size([2, 2])

if __name__ == '__main__':
    test_attentivefp_predictor()
    test_mlp_predictor()
    test_gat_predictor()
    test_gcn_predictor()
    test_gin_predictor()
