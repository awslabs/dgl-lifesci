# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import torch
import torch.nn.functional as F

from dgllife.model.gnn import *

def test_graph1():
    """Graph with node features."""
    g = dgl.graph(([0, 0, 1, 0, 1, 2],
                   [1, 2, 2, 0, 1, 2]), idtype=torch.int32)
    return g, torch.arange(g.num_nodes()).float().reshape(-1, 1)

def test_graph2():
    """Batched graph with node features."""
    g1 = dgl.graph(([0, 0, 1, 0, 1, 2],
                    [1, 2, 2, 0, 1, 2]), idtype=torch.int32)
    g2 = dgl.graph(([0, 1, 1, 1, 0, 1, 2, 3, 4],
                    [1, 2, 3, 4, 0, 1, 2, 3, 4]), idtype=torch.int32)
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.num_nodes()).float().reshape(-1, 1)

def test_graph3():
    """Graph with node and edge features."""
    g = dgl.graph(([0, 0, 1, 0, 1, 2], [1, 2, 2, 0, 1, 2]), idtype=torch.int32)
    return g, torch.arange(g.num_nodes()).float().reshape(-1, 1), \
           torch.arange(2 * g.num_edges()).float().reshape(-1, 2)

def test_graph4():
    """Batched graph with node and edge features."""
    g1 = dgl.graph(([0, 0, 1], [1, 2, 2]), idtype=torch.int32)
    g2 = dgl.graph(([0, 1, 1, 1], [1, 2, 3, 4]), idtype=torch.int32)
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.num_nodes()).float().reshape(-1, 1), \
           torch.arange(2 * bg.num_edges()).float().reshape(-1, 2)

def test_graph5():
    """Graph with node types and edge distances."""
    g1 = dgl.graph(([0, 0, 1], [1, 2, 2]), idtype=torch.int32)
    return g1, torch.LongTensor([0, 1, 0]), torch.randn(3, 1)

def test_graph6():
    """Batched graph with node types and edge distances."""
    g1 = dgl.graph(([0, 0, 1], [1, 2, 2]), idtype=torch.int32)
    g2 = dgl.graph(([0, 1, 1, 1], [1, 2, 3, 4]), idtype=torch.int32)
    bg = dgl.batch([g1, g2])
    return bg, torch.LongTensor([0, 1, 0, 2, 0, 3, 4, 4]), torch.randn(7, 1)

def test_graph7():
    """Graph with categorical node and edge features."""
    g1 = dgl.graph(([0, 0, 1], [1, 2, 2]), idtype=torch.int32)
    return g1, torch.LongTensor([0, 1, 0]), torch.LongTensor([2, 3, 4]), \
           torch.LongTensor([0, 0, 1]), torch.LongTensor([2, 3, 2])

def test_graph8():
    """Batched graph with categorical node and edge features."""
    g1 = dgl.graph(([0, 0, 1], [1, 2, 2]), idtype=torch.int32)
    g2 = dgl.graph(([0, 1, 1, 1], [1, 2, 3, 4]), idtype=torch.int32)
    bg = dgl.batch([g1, g2])
    return bg, torch.LongTensor([0, 1, 0, 2, 1, 0, 2, 2]), \
           torch.LongTensor([2, 3, 4, 1, 0, 1, 2, 2]), \
           torch.LongTensor([0, 0, 1, 2, 1, 0, 0]), \
           torch.LongTensor([2, 3, 2, 0, 1, 2, 1])

def test_graph9():
    """Batched graph with categorical node features and continuous edge features"""
    g1 = dgl.graph(([0, 0, 1], [1, 2, 2]), idtype=torch.int32)
    g2 = dgl.graph(([0, 1, 1, 1], [1, 2, 3, 4]), idtype=torch.int32)
    bg = dgl.batch([g1, g2])
    return bg, torch.zeros(bg.num_nodes()).long(), \
           torch.randn(bg.num_edges(), 2).float()

def test_graph10():
    """Graph with node and edge features, but no edges."""
    g = dgl.graph(([], []), num_nodes=3, idtype=torch.int32)
    return g, torch.arange(g.num_nodes()).float().reshape(-1, 1), \
           torch.arange(2 * g.num_edges()).float().reshape(-1, 2)

def test_attentivefp():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats, edge_feats = test_graph3()
    g, node_feats, edge_feats = g.to(device), node_feats.to(device), edge_feats.to(device)
    bg, batch_node_feats, batch_edge_feats = test_graph4()
    bg, batch_node_feats, batch_edge_feats = bg.to(device), batch_node_feats.to(device), \
                                             batch_edge_feats.to(device)

    # Test AttentiveFPGNN
    gnn = AttentiveFPGNN(node_feat_size=1,
                         edge_feat_size=2,
                         num_layers=1,
                         graph_feat_size=1,
                         dropout=0.).to(device)
    gnn.reset_parameters()
    assert gnn(g, node_feats, edge_feats).shape == torch.Size([3, 1])
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == torch.Size([8, 1])

def test_gat():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    # Test default setting
    gnn = GAT(in_feats=1).to(device)
    gnn.reset_parameters()
    assert gnn(g, node_feats).shape == torch.Size([3, 32])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 32])

    # Test configured setting
    gnn = GAT(in_feats=1,
              hidden_feats=[1, 1],
              num_heads=[2, 3],
              feat_drops=[0.1, 0.1],
              attn_drops=[0.1, 0.1],
              alphas=[0.2, 0.2],
              residuals=[True, True],
              agg_modes=['flatten', 'mean'],
              activations=[None, F.elu]).to(device)
    assert gnn(g, node_feats).shape == torch.Size([3, 1])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 1])

    gnn = GAT(in_feats=1,
              hidden_feats=[1, 1],
              num_heads=[2, 3],
              feat_drops=[0.1, 0.1],
              attn_drops=[0.1, 0.1],
              alphas=[0.2, 0.2],
              residuals=[True, True],
              agg_modes=['mean', 'flatten'],
              activations=[None, F.elu]).to(device)
    assert gnn(g, node_feats).shape == torch.Size([3, 3])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 3])

def test_gatv2():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    # Test default setting
    gnn = GATv2(in_feats=1).to(device)
    gnn.reset_parameters()
    assert gnn(g, node_feats).shape == torch.Size([3, 32])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 32])

    # Test configured setting
    gnn = GATv2(in_feats=1,
                hidden_feats=[1, 1],
                num_heads=[2, 3],
                feat_drops=[0.1, 0.1],
                attn_drops=[0.1, 0.1],
                alphas=[0.2, 0.2],
                residuals=[True, True],
                agg_modes=['flatten', 'mean'],
                activations=[None, F.elu]).to(device)
    assert gnn(g, node_feats).shape == torch.Size([3, 1])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 1])

    gnn = GATv2(in_feats=1,
                hidden_feats=[1, 1],
                num_heads=[2, 3],
                feat_drops=[0.1, 0.1],
                attn_drops=[0.1, 0.1],
                alphas=[0.2, 0.2],
                residuals=[True, True],
                agg_modes=['mean', 'flatten'],
                activations=[None, F.elu],
                share_weights=[True, False]).to(device)
    assert gnn(g, node_feats).shape == torch.Size([3, 3])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 3])

def test_gatv2_get_attention():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    # Test default setting, get_attention=True
    gnn = GATv2(in_feats=1).to(device)
    gnn.reset_parameters()
    out_feats, out_attentions = gnn(g, node_feats, get_attention=True)
    assert out_feats.shape == torch.Size([3, 32])
    len(out_attentions) == 2
    assert out_attentions[0].shape == torch.Size([6, 4, 1])
    assert out_attentions[1].shape == torch.Size([6, 4, 1])
    batch_out_feats, batch_out_attentions = gnn(
        bg, batch_node_feats, get_attention=True
    )
    assert batch_out_feats.shape == torch.Size([8, 32])
    len(batch_out_attentions) == 2
    assert batch_out_attentions[0].shape == torch.Size([15, 4, 1])
    assert batch_out_attentions[1].shape == torch.Size([15, 4, 1])

    # Test configured setting, get_attention=True
    gnn = GATv2(in_feats=1,
                hidden_feats=[1, 1],
                num_heads=[2, 3],
                feat_drops=[0.1, 0.1],
                attn_drops=[0.1, 0.1],
                alphas=[0.2, 0.2],
                residuals=[True, True],
                agg_modes=['flatten', 'mean'],
                activations=[None, F.elu]).to(device)
    out_feats, out_attentions = gnn(g, node_feats, get_attention=True)
    assert out_feats.shape == torch.Size([3, 1])
    len(out_attentions) == 2
    assert out_attentions[0].shape == torch.Size([6, 2, 1])
    assert out_attentions[1].shape == torch.Size([6, 3, 1])
    batch_out_feats, batch_out_attentions = gnn(
        bg, batch_node_feats, get_attention=True
    )
    assert batch_out_feats.shape == torch.Size([8, 1])
    assert len(batch_out_attentions) == 2
    assert batch_out_attentions[0].shape == torch.Size([15, 2, 1])
    assert batch_out_attentions[1].shape == torch.Size([15, 3, 1])

    gnn = GATv2(in_feats=1,
                hidden_feats=[1, 1],
                num_heads=[2, 3],
                feat_drops=[0.1, 0.1],
                attn_drops=[0.1, 0.1],
                alphas=[0.2, 0.2],
                residuals=[True, True],
                agg_modes=['mean', 'flatten'],
                activations=[None, F.elu],
                share_weights=[True, False]).to(device)
    assert gnn(g, node_feats).shape == torch.Size([3, 3])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 3])
    out_feats, out_attentions = gnn(g, node_feats, get_attention=True)
    assert out_feats.shape == torch.Size([3, 3])
    len(out_attentions) == 2
    assert out_attentions[0].shape == torch.Size([6, 2, 1])
    assert out_attentions[1].shape == torch.Size([6, 3, 1])
    batch_out_feats, batch_out_attentions = gnn(
        bg, batch_node_feats, get_attention=True
    )
    assert batch_out_feats.shape == torch.Size([8, 3])
    assert len(batch_out_attentions) == 2
    assert batch_out_attentions[0].shape == torch.Size([15, 2, 1])
    assert batch_out_attentions[1].shape == torch.Size([15, 3, 1])

def test_gcn():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    # Test default setting
    gnn = GCN(in_feats=1).to(device)
    gnn.reset_parameters()
    assert gnn(g, node_feats).shape == torch.Size([3, 64])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 64])

    # Test configured setting
    gnn = GCN(in_feats=1,
              hidden_feats=[1, 1],
              activation=[F.relu, F.relu],
              residual=[True, True],
              batchnorm=[True, True],
              dropout=[0.2, 0.2]).to(device)
    assert gnn(g, node_feats).shape == torch.Size([3, 1])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 1])

def test_gin():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats1, node_feats2, edge_feats1, edge_feats2 = test_graph7()
    g = g.to(device)
    node_feats1, node_feats2 = node_feats1.to(device), node_feats2.to(device)
    edge_feats1, edge_feats2 = edge_feats1.to(device), edge_feats2.to(device)
    bg, batch_node_feats1, batch_node_feats2, \
    batch_edge_feats1, batch_edge_feats2 = test_graph8()
    bg = bg.to(device)
    batch_node_feats1, batch_node_feats2 = batch_node_feats1.to(device), \
                                           batch_node_feats2.to(device)
    batch_edge_feats1, batch_edge_feats2 = batch_edge_feats1.to(device), \
                                           batch_edge_feats2.to(device)

    # Test default setting
    gnn = GIN(num_node_emb_list=[3, 5], num_edge_emb_list=[3, 4]).to(device)
    gnn.reset_parameters()
    assert gnn(g, [node_feats1, node_feats2], [edge_feats1, edge_feats2]).shape \
           == torch.Size([3, 300])
    assert gnn(bg, [batch_node_feats1, batch_node_feats2],
               [batch_edge_feats1, batch_edge_feats2]).shape == torch.Size([8, 300])

    # Test configured setting
    gnn = GIN(num_node_emb_list=[3, 5], num_edge_emb_list=[3, 4],
              num_layers=2, emb_dim=10, JK='concat', dropout=0.1).to(device)
    assert gnn(g, [node_feats1, node_feats2], [edge_feats1, edge_feats2]).shape \
           == torch.Size([3, 30])
    assert gnn(bg, [batch_node_feats1, batch_node_feats2],
               [batch_edge_feats1, batch_edge_feats2]).shape == torch.Size([8, 30])

def test_mgcn():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_types, edge_dists = test_graph5()
    g, node_types, edge_dists = g.to(device), node_types.to(device), edge_dists.to(device)
    bg, batch_node_types, batch_edge_dists = test_graph6()
    bg, batch_node_types, batch_edge_dists = bg.to(device), batch_node_types.to(device), \
                                             batch_edge_dists.to(device)

    # Test default setting
    gnn = MGCNGNN().to(device)
    gnn.reset_parameters()
    assert gnn(g, node_types, edge_dists).shape == torch.Size([3, 512])
    assert gnn(bg, batch_node_types, batch_edge_dists).shape == torch.Size([8, 512])

    # Test configured setting
    gnn = MGCNGNN(feats=2,
                  n_layers=2,
                  num_node_types=5,
                  num_edge_types=150,
                  cutoff=0.3).to(device)
    assert gnn(g, node_types, edge_dists).shape == torch.Size([3, 6])
    assert gnn(bg, batch_node_types, batch_edge_dists).shape == torch.Size([8, 6])

def test_mpnn():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats, edge_feats = test_graph3()
    g, node_feats, edge_feats = g.to(device), node_feats.to(device), edge_feats.to(device)
    bg, batch_node_feats, batch_edge_feats = test_graph4()
    bg, batch_node_feats, batch_edge_feats = bg.to(device), batch_node_feats.to(device), \
                                             batch_edge_feats.to(device)

    # Test default setting
    gnn = MPNNGNN(node_in_feats=1,
                  edge_in_feats=2).to(device)
    gnn.reset_parameters()
    assert gnn(g, node_feats, edge_feats).shape == torch.Size([3, 64])
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == torch.Size([8, 64])

    # Test configured setting
    gnn = MPNNGNN(node_in_feats=1,
                  edge_in_feats=2,
                  node_out_feats=2,
                  edge_hidden_feats=2,
                  num_step_message_passing=2).to(device)
    assert gnn(g, node_feats, edge_feats).shape == torch.Size([3, 2])
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == torch.Size([8, 2])

def test_schnet():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_types, edge_dists = test_graph5()
    g, node_types, edge_dists = g.to(device), node_types.to(device), edge_dists.to(device)
    bg, batch_node_types, batch_edge_dists = test_graph6()
    bg, batch_node_types, batch_edge_dists = bg.to(device), batch_node_types.to(device), \
                                             batch_edge_dists.to(device)

    # Test default setting
    gnn = SchNetGNN().to(device)
    gnn.reset_parameters()
    assert gnn(g, node_types, edge_dists).shape == torch.Size([3, 64])
    assert gnn(bg, batch_node_types, batch_edge_dists).shape == torch.Size([8, 64])

    # Test configured setting
    gnn = SchNetGNN(num_node_types=5,
                    node_feats=2,
                    hidden_feats=[3],
                    cutoff=0.3).to(device)
    assert gnn(g, node_types, edge_dists).shape == torch.Size([3, 2])
    assert gnn(bg, batch_node_types, batch_edge_dists).shape == torch.Size([8, 2])

def test_weave():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats, edge_feats = test_graph3()
    g, node_feats, edge_feats = g.to(device), node_feats.to(device), edge_feats.to(device)
    bg, batch_node_feats, batch_edge_feats = test_graph4()
    bg, batch_node_feats, batch_edge_feats = bg.to(device), batch_node_feats.to(device), \
                                             batch_edge_feats.to(device)

    # Test default setting
    gnn = WeaveGNN(node_in_feats=1,
                   edge_in_feats=2).to(device)
    gnn.reset_parameters()
    assert gnn(g, node_feats, edge_feats).shape == torch.Size([3, 50])
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == torch.Size([8, 50])

    # Test configured setting
    gnn = WeaveGNN(node_in_feats=1,
                   edge_in_feats=2,
                   num_layers=1,
                   hidden_feats=2).to(device)
    assert gnn(g, node_feats, edge_feats).shape == torch.Size([3, 2])
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == torch.Size([8, 2])

def test_wln():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats, edge_feats = test_graph3()
    g, node_feats, edge_feats = g.to(device), node_feats.to(device), edge_feats.to(device)
    g_ne, node_feats_ne, edge_feats_ne = test_graph10()
    g_ne, node_feats_ne, edge_feats_ne = g_ne.to(device), node_feats_ne.to(device), edge_feats_ne.to(device)
    bg, batch_node_feats, batch_edge_feats = test_graph4()
    bg, batch_node_feats, batch_edge_feats = bg.to(device), batch_node_feats.to(device), \
                                             batch_edge_feats.to(device)

    # Test default setting
    gnn = WLN(node_in_feats=1,
              edge_in_feats=2).to(device)
    gnn.reset_parameters()
    assert gnn(g, node_feats, edge_feats).shape == torch.Size([3, 300])
    assert gnn(g_ne, node_feats_ne, edge_feats_ne).shape == torch.Size([3, 300])
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == torch.Size([8, 300])

    # Test configured setting
    gnn = WLN(node_in_feats=1,
              edge_in_feats=2,
              node_out_feats=3,
              n_layers=1).to(device)
    assert gnn(g, node_feats, edge_feats).shape == torch.Size([3, 3])
    assert gnn(g_ne, node_feats_ne, edge_feats_ne).shape == torch.Size([3, 3])
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == torch.Size([8, 3])

def test_graphsage():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    # Test default setting
    gnn = GraphSAGE(in_feats=1).to(device)
    gnn.reset_parameters()
    assert gnn(g, node_feats).shape == torch.Size([3, 64])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 64])

    # Test configured setting
    gnn = GraphSAGE(in_feats=1,
                    hidden_feats=[1, 1],
                    activation=[F.relu, F.relu],
                    dropout=[0.2, 0.2],
                    aggregator_type=['gcn', 'gcn']).to(device)
    assert gnn(g, node_feats).shape == torch.Size([3, 1])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 1])

    gnn.reset_parameters()
    assert gnn(g, node_feats).shape == torch.Size([3, 1])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 1])

def test_gnn_ogb():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    bg, batch_node_feats, batch_edge_feats = test_graph9()
    bg, batch_node_feats, batch_edge_feats = bg.to(device), batch_node_feats.to(device), \
                                             batch_edge_feats.to(device)

    # Test default setting
    gnn = GNNOGB(in_edge_feats=batch_edge_feats.shape[-1],
                 hidden_feats=2).to(device)
    gnn.reset_parameters()
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == \
           torch.Size([bg.num_nodes(), 2])

    # Test configured setting
    gnn = GNNOGB(in_edge_feats=batch_edge_feats.shape[-1],
                 num_node_types=2,
                 hidden_feats=2,
                 n_layers=2,
                 batchnorm=False,
                 activation=None,
                 dropout=0.1,
                 gnn_type='gin',
                 virtual_node=False,
                 residual=True,
                 jk=True).to(device)
    gnn.reset_parameters()
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == \
           torch.Size([bg.num_nodes(), 2])

def test_nf():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    # Test default setting
    gnn = NFGNN(in_feats=1).to(device)
    gnn.reset_parameters()
    assert gnn(g, node_feats).shape == torch.Size([3, 64])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 64])

    # Test configured setting
    gnn = NFGNN(in_feats=1,
                hidden_feats=[2, 2, 2],
                max_degree=5,
                activation=[None, None, None],
                batchnorm=[False, False, False],
                dropout=[0.5, 0.5, 0.5]).to(device)
    gnn.reset_parameters()
    assert gnn(g, node_feats).shape == torch.Size([3, 2])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 2])

def test_pagtn():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats, edge_feats = test_graph3()
    g, node_feats, edge_feats = g.to(device), node_feats.to(device), edge_feats.to(device)
    bg, batch_node_feats, batch_edge_feats = test_graph4()
    bg, batch_node_feats, batch_edge_feats = bg.to(device), batch_node_feats.to(device), \
                                             batch_edge_feats.to(device)
    gnn = PAGTNGNN(node_in_feats = 1,
                   node_out_feats = 2,
                   node_hid_feats = 20,
                   edge_feats = 2).to(device)
    assert gnn(g, node_feats, edge_feats).shape == torch.Size([3, 2])
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == torch.Size([8, 2])

if __name__ == '__main__':
    test_attentivefp()
    test_gat()
    test_gcn()
    test_gin()
    test_mgcn()
    test_mpnn()
    test_schnet()
    test_weave()
    test_wln()
    test_gnn_ogb()
    test_graphsage()
    test_nf()
    test_pagtn()
