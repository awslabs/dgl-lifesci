# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import torch
import torch.nn.functional as F

from dgl import DGLGraph

from dgllife.model.model_zoo.gcn_link_predictor import *
from dgllife.model.model_zoo.sage_link_predictor import *

def test_graph1():
    """Graph with node features."""
    g = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g, torch.arange(g.number_of_nodes()).float().reshape(-1, 1)

def test_gcn_link_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)

    # Test configured setting
    gcn_model = GCN(
        in_feats=node_feats.size(-1),
        n_hidden=2,
        out_feats=2,
        num_layers=2,
        dropout=0.1).to(device)
    gcn_model.train()
    assert gcn_model(g, node_feats).shape == torch.Size([1, 2])

    gcn_link_predictor = GCNLinkPredictor(
        in_channels=2,
        hidden_channels=2,
        num_layers=2,
        dropout=0.1).to(device)
    assert gcn_link_predictor(g, node_feats).shape == torch.Size([1, 1])

def test_sage_link_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)

    # Test configured setting
    sage_model = SAGE(
        in_feats=node_feats.size(-1),
        n_hidden=2,
        out_feats=2,
        num_layers=2,
        dropout=0.1).to(device)
    sage_model.train()
    assert sage_model(g, node_feats).shape == torch.Size([1, 2])

    sage_link_predictor = SAGELinkPredictor(
        in_channels=2,
        hidden_channels=2,
        num_layers=2,
        dropout=0.1).to(device)
    assert sage_link_predictor(g, node_feats).shape == torch.Size([1, 1])

if __name__ == '__main__':
    test_gcn_link_predictor()
    test_sage_link_predictor()
