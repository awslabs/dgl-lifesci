# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import torch
import torch.nn.functional as F

from dgllife.model.readout import *

def test_graph1():
    """Graph with node features"""
    g = dgl.graph(([0, 0, 1], [1, 2, 2]))
    return g, torch.arange(g.num_nodes()).float().reshape(-1, 1)

def test_graph2():
    "Batched graph with node features"
    g1 = dgl.graph(([0, 0, 1], [1, 2, 2]))
    g2 = dgl.graph(([0, 1, 1, 1], [1, 2, 3, 4]))
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.num_nodes()).float().reshape(-1, 1)

def test_attentive_fp_readout():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)
    model = AttentiveFPReadout(feat_size=1,
                               num_timesteps=1).to(device)
    assert model(g, node_feats).shape == torch.Size([1, 1])
    assert model(bg, batch_node_feats).shape == torch.Size([2, 1])

def test_mlp_readout():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    model = MLPNodeReadout(node_feats=1,
                            hidden_feats=2,
                            graph_feats=3,
                            activation=F.relu,
                            mode='sum').to(device)
    assert model(g, node_feats).shape == torch.Size([1, 3])
    assert model(bg, batch_node_feats).shape == torch.Size([2, 3])

    model = MLPNodeReadout(node_feats=1,
                            hidden_feats=2,
                            graph_feats=3,
                            mode='max').to(device)
    assert model(g, node_feats).shape == torch.Size([1, 3])
    assert model(bg, batch_node_feats).shape == torch.Size([2, 3])

    model = MLPNodeReadout(node_feats=1,
                            hidden_feats=2,
                            graph_feats=3,
                            mode='mean').to(device)
    assert model(g, node_feats).shape == torch.Size([1, 3])
    assert model(bg, batch_node_feats).shape == torch.Size([2, 3])

def test_weave_readout():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    model = WeaveGather(node_in_feats=1).to(device)
    assert model(g, node_feats).shape == torch.Size([1, 1])
    assert model(bg, batch_node_feats).shape == torch.Size([2, 1])

    model = WeaveGather(node_in_feats=1, gaussian_expand=False).to(device)
    assert model(g, node_feats).shape == torch.Size([1, 1])
    assert model(bg, batch_node_feats).shape == torch.Size([2, 1])

def test_weighted_sum_and_max():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)
    model = WeightedSumAndMax(in_feats=1).to(device)
    assert model(g, node_feats).shape == torch.Size([1, 2])
    assert model(bg, batch_node_feats).shape == torch.Size([2, 2])

def test_sum_and_max():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)
    model = SumAndMax()
    assert model(g, node_feats).shape == torch.Size([1, 2])
    assert model(bg, batch_node_feats).shape == torch.Size([2, 2])

if __name__ == '__main__':
    test_attentive_fp_readout()
    test_mlp_readout()
    test_weave_readout()
    test_weighted_sum_and_max()
    test_sum_and_max()
