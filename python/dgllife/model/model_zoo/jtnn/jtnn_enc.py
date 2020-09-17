# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=C0111, C0103, E1101, W0611, W0612, W0221

import numpy as np
import torch
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl import bfs_edges_generator

from .nnutils import GRUUpdate

def level_order(forest, roots):
    device = forest.device
    edges = list(bfs_edges_generator(forest, roots))
    if len(edges) == 0:
        # no edges in the tree; do not perform loopy BP
        return
    edges = [e.to(device) for e in edges]
    _, leaves = forest.find_edges(edges[-1])
    edges_back = list(bfs_edges_generator(forest, roots, reverse=True))
    edges_back = [e.to(device) for e in edges_back]
    yield from reversed(edges_back)
    yield from edges

class EncoderGatherUpdate(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size

        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.W.reset_parameters()

    def forward(self, nodes):
        x = nodes.data['x']
        m = nodes.data['m']
        return {
            'h': torch.relu(self.W(torch.cat([x, m], 1))),
        }

class DGLJTNNEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, embedding):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        self.embedding = embedding
        self.enc_tree_update = GRUUpdate(hidden_size)
        self.enc_tree_gather_update = EncoderGatherUpdate(hidden_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.embedding.reset_parameters()
        self.enc_tree_update.reset_parameters()
        self.enc_tree_gather_update.reset_parameters()

    def forward(self, mol_trees):
        mol_tree_batch = dgl.batch(mol_trees)

        # Build line graph to prepare for belief propagation
        mol_tree_batch_lg = dgl.line_graph(mol_tree_batch, backtracking=False, shared=True)

        return self.run(mol_tree_batch, mol_tree_batch_lg)

    def run(self, mol_tree_batch, mol_tree_batch_lg):
        device = mol_tree_batch.device

        # Since tree roots are designated to 0.  In the batched graph we can
        # simply find the corresponding node ID by looking at node_offset
        node_offset = np.cumsum(np.insert(mol_tree_batch.batch_num_nodes().cpu().numpy(), 0, 0))
        root_ids = torch.tensor(node_offset[:-1], device=device, dtype=mol_tree_batch.idtype)
        n_nodes = mol_tree_batch.num_nodes()
        n_edges = mol_tree_batch.num_edges()

        # Assign structure embeddings to tree nodes
        mol_tree_batch.ndata.update({
            'x': self.embedding(mol_tree_batch.ndata['wid']),
            'm': torch.zeros(n_nodes, self.hidden_size).to(device),
            'h': torch.zeros(n_nodes, self.hidden_size).to(device)
        })

        # Initialize the intermediate variables according to Eq (4)-(8).
        # Also initialize the src_x and dst_x fields.
        # TODO: context?
        mol_tree_batch.edata.update({
            's': torch.zeros(n_edges, self.hidden_size).to(device),
            'm': torch.zeros(n_edges, self.hidden_size).to(device),
            'r': torch.zeros(n_edges, self.hidden_size).to(device),
            'z': torch.zeros(n_edges, self.hidden_size).to(device),
            'src_x': torch.zeros(n_edges, self.hidden_size).to(device),
            'dst_x': torch.zeros(n_edges, self.hidden_size).to(device),
            'rm': torch.zeros(n_edges, self.hidden_size).to(device),
            'accum_rm': torch.zeros(n_edges, self.hidden_size).to(device),
        })

        # Send the source/destination node features to edges
        mol_tree_batch.apply_edges(
            func=lambda edges: {
                'src_x': edges.src['x'], 'dst_x': edges.dst['x']},
        )

        # Message passing
        # I exploited the fact that the reduce function is a sum of incoming
        # messages, and the uncomputed messages are zero vectors.  Essentially,
        # we can always compute s_ij as the sum of incoming m_ij, no matter
        # if m_ij is actually computed or not.
        mol_tree_batch_lg.ndata.update(mol_tree_batch.edata)
        for eid in level_order(mol_tree_batch, root_ids):
            eid = eid.to(mol_tree_batch_lg.device)
            mol_tree_batch_lg.pull(eid, fn.copy_u('m', 'm'), fn.sum('m', 's'))
            mol_tree_batch_lg.pull(eid, fn.copy_u('rm', 'rm'), fn.sum('rm', 'rm'))
            mol_tree_batch_lg.apply_nodes(self.enc_tree_update)

        # Readout
        mol_tree_batch.edata.update(mol_tree_batch_lg.ndata)
        mol_tree_batch.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'm'))
        mol_tree_batch.apply_nodes(self.enc_tree_gather_update)

        root_vecs = mol_tree_batch.nodes[root_ids].data['h']

        return mol_tree_batch, root_vecs
