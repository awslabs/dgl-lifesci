# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import numpy as np
import torch

from dgllife.model.model_zoo import *

def get_complete_graph(num_nodes):
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            src.append(i)
            dst.append(j)
    return dgl.graph((src, dst))

def test_graph1():
    """
    Bi-directed graphs and complete graphs for the molecules.
    In addition to node features/edge features, we also return
    features for the pairs of nodes.
    """
    mol_graph = dgl.graph(([0, 0, 1], [1, 2, 2]))
    node_feats = torch.arange(mol_graph.num_nodes()).float().reshape(-1, 1)
    edge_feats = torch.arange(2 * mol_graph.num_edges()).float().reshape(-1, 2)

    complete_graph = get_complete_graph(mol_graph.num_nodes())
    atom_pair_feats = torch.arange(complete_graph.num_edges()).float().reshape(-1, 1)

    return mol_graph, node_feats, edge_feats, complete_graph, atom_pair_feats

def test_graph2():
    """Batched version of test_graph1"""
    mol_graph1 = dgl.graph(([0, 0, 1], [1, 2, 2]))
    mol_graph2 = dgl.graph(([0, 1, 1, 1], [1, 2, 3, 4]))
    batch_mol_graph = dgl.batch([mol_graph1, mol_graph2])
    node_feats = torch.arange(batch_mol_graph.num_nodes()).float().reshape(-1, 1)
    edge_feats = torch.arange(2 * batch_mol_graph.num_edges()).float().reshape(-1, 2)

    complete_graph1 = get_complete_graph(mol_graph1.num_nodes())
    complete_graph2 = get_complete_graph(mol_graph2.num_nodes())
    batch_complete_graph = dgl.batch([complete_graph1, complete_graph2])
    atom_pair_feats = torch.arange(batch_complete_graph.num_edges()).float().reshape(-1, 1)

    return batch_mol_graph, node_feats, edge_feats, batch_complete_graph, atom_pair_feats

def test_wln_reaction_center():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    mol_graph, node_feats, edge_feats, complete_graph, atom_pair_feats = test_graph1()
    mol_graph = mol_graph.to(device)
    node_feats, edge_feats = node_feats.to(device), edge_feats.to(device)
    complete_graph = complete_graph.to(device)
    atom_pair_feats = atom_pair_feats.to(device)

    batch_mol_graph, batch_node_feats, batch_edge_feats, batch_complete_graph, \
    batch_atom_pair_feats = test_graph2()
    batch_mol_graph = batch_mol_graph.to(device)
    batch_node_feats, batch_edge_feats = batch_node_feats.to(device), batch_edge_feats.to(device)
    batch_complete_graph = batch_complete_graph.to(device)
    batch_atom_pair_feats = batch_atom_pair_feats.to(device)

    # Test default setting
    model = WLNReactionCenter(node_in_feats=1,
                              edge_in_feats=2,
                              node_pair_in_feats=1).to(device)
    assert model(mol_graph, complete_graph, node_feats, edge_feats, atom_pair_feats)[0].shape == \
           torch.Size([complete_graph.num_edges(), 5])
    assert model(batch_mol_graph, batch_complete_graph, batch_node_feats,
                 batch_edge_feats, batch_atom_pair_feats)[0].shape == \
           torch.Size([batch_complete_graph.num_edges(), 5])

    # Test configured setting
    model = WLNReactionCenter(node_in_feats=1,
                              edge_in_feats=2,
                              node_pair_in_feats=1,
                              node_out_feats=1,
                              n_layers=1,
                              n_tasks=1).to(device)
    assert model(mol_graph, complete_graph, node_feats, edge_feats, atom_pair_feats)[0].shape == \
           torch.Size([complete_graph.num_edges(), 1])
    assert model(batch_mol_graph, batch_complete_graph, batch_node_feats,
                 batch_edge_feats, batch_atom_pair_feats)[0].shape == \
           torch.Size([batch_complete_graph.num_edges(), 1])

def test_reactant_product_graph1():
    edges = (np.array([0, 1, 2]), np.array([1, 2, 2]))
    reactant_g = dgl.graph(edges)
    reactant_node_feats = torch.arange(
        reactant_g.num_nodes()).float().reshape(-1, 1)
    reactant_edge_feats = torch.arange(
        reactant_g.num_edges()).float().reshape(-1, 1)

    product_g = []
    batch_num_candidate_products = []
    for i in range(1, 2):
        product_g.extend([
            dgl.graph(edges) for _ in range(i)
        ])
        batch_num_candidate_products.append(i)
    product_g = dgl.batch(product_g)
    product_node_feats = torch.arange(
        product_g.num_nodes()).float().reshape(-1, 1)
    product_edge_feats = torch.arange(
        product_g.num_edges()).float().reshape(-1, 1)
    product_scores = torch.randn(sum(batch_num_candidate_products), 1)

    return reactant_g, reactant_node_feats, reactant_edge_feats, product_g, product_node_feats, \
           product_edge_feats, product_scores, batch_num_candidate_products

def test_reactant_product_graph2():
    batch_size = 2
    edges = (np.array([0, 1, 2]), np.array([1, 2, 2]))
    reactant_g = []
    for _ in range(batch_size):
        reactant_g.append(dgl.graph(edges))
    reactant_g = dgl.batch(reactant_g)
    reactant_node_feats = torch.arange(
        reactant_g.num_nodes()).float().reshape(-1, 1)
    reactant_edge_feats = torch.arange(
        reactant_g.num_edges()).float().reshape(-1, 1)

    product_g = []
    batch_num_candidate_products = []
    for i in range(1, batch_size + 1):
        product_g.extend([
            dgl.graph(edges) for _ in range(i)
        ])
        batch_num_candidate_products.append(i)
    product_g = dgl.batch(product_g)
    product_node_feats = torch.arange(
        product_g.num_nodes()).float().reshape(-1, 1)
    product_edge_feats = torch.arange(
        product_g.num_edges()).float().reshape(-1, 1)
    product_scores = torch.randn(sum(batch_num_candidate_products), 1)

    return reactant_g, reactant_node_feats, reactant_edge_feats, product_g, product_node_feats, \
           product_edge_feats, product_scores, batch_num_candidate_products

def test_wln_candidate_ranking():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    reactant_g, reactant_node_feats, reactant_edge_feats, product_g, product_node_feats, \
    product_edge_feats, product_scores, num_candidate_products = test_reactant_product_graph1()
    reactant_g = reactant_g.to(device)
    product_g = product_g.to(device)
    reactant_node_feats, reactant_edge_feats = reactant_node_feats.to(device), reactant_edge_feats.to(device)
    product_node_feats, product_edge_feats, product_scores = product_node_feats.to(device), \
                                                             product_edge_feats.to(device), \
                                                             product_scores.to(device)

    batch_reactant_g, batch_reactant_node_feats, batch_reactant_edge_feats, batch_product_g, \
    batch_product_node_feats, batch_product_edge_feats, batch_product_scores, \
    batch_num_candidate_products = test_reactant_product_graph2()
    batch_reactant_g = batch_reactant_g.to(device)
    batch_product_g = batch_product_g.to(device)
    batch_reactant_node_feats = batch_reactant_node_feats.to(device)
    batch_reactant_edge_feats = batch_reactant_edge_feats.to(device)
    batch_product_node_feats = batch_product_node_feats.to(device)
    batch_product_edge_feats = batch_product_edge_feats.to(device)
    batch_product_scores = batch_product_scores.to(device)

    # Test default setting
    model = WLNReactionRanking(node_in_feats=1,
                               edge_in_feats=1).to(device)
    assert model(reactant_g, reactant_node_feats, reactant_edge_feats, product_g,
                 product_node_feats, product_edge_feats, product_scores,
                 num_candidate_products).shape == torch.Size([sum(num_candidate_products), 1])
    assert model(batch_reactant_g, batch_reactant_node_feats, batch_reactant_edge_feats,
                 batch_product_g, batch_product_node_feats, batch_product_edge_feats,
                 batch_product_scores, batch_num_candidate_products).shape == \
           torch.Size([sum(batch_num_candidate_products), 1])

    model = WLNReactionRanking(node_in_feats=1,
                               edge_in_feats=1,
                               node_hidden_feats=100,
                               num_encode_gnn_layers=2).to(device)
    assert model(reactant_g, reactant_node_feats, reactant_edge_feats, product_g,
                 product_node_feats, product_edge_feats, product_scores,
                 num_candidate_products).shape == torch.Size([sum(num_candidate_products), 1])
    assert model(batch_reactant_g, batch_reactant_node_feats, batch_reactant_edge_feats,
                 batch_product_g, batch_product_node_feats, batch_product_edge_feats,
                 batch_product_scores, batch_num_candidate_products).shape == \
           torch.Size([sum(batch_num_candidate_products), 1])

if __name__ == '__main__':
    test_wln_reaction_center()
    test_wln_candidate_ranking()
