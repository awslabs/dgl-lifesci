# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=C0111, C0103, E1101, W0611, W0612, W1508, I1101, W0221
# pylint: disable=redefined-outer-name

import os
import torch
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl import mean_nodes

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5

PAPER = os.getenv('PAPER', False)

class LoopyBPUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(LoopyBPUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.W_h.reset_parameters()

    def forward(self, node):
        msg_input = node.data['msg_input']
        msg_delta = self.W_h(node.data['accum_msg'] + node.data['alpha'])
        msg = torch.relu(msg_input + msg_delta)
        return {'msg': msg}

class GatherUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(GatherUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.W_o.reset_parameters()

    def forward(self, node):
        if PAPER:
            m = node.data['m'] + node.data['accum_alpha']
        else:
            m = node.data['m'] + node.data['alpha']
        return {
            'h': torch.relu(self.W_o(torch.cat([node.data['x'], m], 1))),
        }

class DGLJTMPN(nn.Module):
    def __init__(self, hidden_size, depth):
        nn.Module.__init__(self)

        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)

        self.loopy_bp_updater = LoopyBPUpdate(hidden_size)
        self.gather_updater = GatherUpdate(hidden_size)
        self.hidden_size = hidden_size

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.W_i.reset_parameters()
        self.loopy_bp_updater.reset_parameters()
        self.gather_updater.reset_parameters()

    def forward(self, cand_batch, mol_tree_batch):
        cand_graphs, tree_mess_src_edges, tree_mess_tgt_edges, tree_mess_tgt_nodes = cand_batch

        cand_line_graph = dgl.line_graph(cand_graphs, backtracking=False, shared=True)

        cand_graphs = self.run(
            cand_graphs, cand_line_graph, tree_mess_src_edges, tree_mess_tgt_edges,
            tree_mess_tgt_nodes, mol_tree_batch)

        g_repr = mean_nodes(cand_graphs, 'h')

        return g_repr

    def run(self, cand_graphs, cand_line_graph, tree_mess_src_edges, tree_mess_tgt_edges,
            tree_mess_tgt_nodes, mol_tree_batch):
        device = cand_graphs.device
        n_nodes = cand_graphs.num_nodes()

        cand_graphs.apply_edges(
            func=lambda edges: {'src_x': edges.src['x']},
        )

        bond_features = cand_line_graph.ndata['x']
        source_features = cand_line_graph.ndata['src_x']
        features = torch.cat([source_features, bond_features], 1)
        msg_input = self.W_i(features)
        cand_line_graph.ndata.update({
            'msg_input': msg_input,
            'msg': torch.relu(msg_input),
            'accum_msg': torch.zeros_like(msg_input),
        })
        zero_node_state = bond_features.new(n_nodes, self.hidden_size).zero_()
        cand_graphs.ndata.update({
            'm': zero_node_state.clone(),
            'h': zero_node_state.clone(),
        })

        cand_graphs.edata['alpha'] = \
            torch.zeros(cand_graphs.num_edges(), self.hidden_size).to(device)
        cand_graphs.ndata['alpha'] = zero_node_state
        if tree_mess_src_edges.shape[0] > 0:
            if PAPER:
                src_u, src_v = tree_mess_src_edges.unbind(1)
                tgt_u, tgt_v = tree_mess_tgt_edges.unbind(1)
                eid = mol_tree_batch.edge_ids(src_u, src_v).long()
                alpha = mol_tree_batch.edata['m'][eid]
                cand_graphs.edges[tgt_u, tgt_v].data['alpha'] = alpha
            else:
                src_u, src_v = tree_mess_src_edges.unbind(1)
                eid = mol_tree_batch.edge_ids(src_u, src_v).long()
                alpha = mol_tree_batch.edata['m'][eid]
                node_idx = (tree_mess_tgt_nodes
                            .to(device=zero_node_state.device, dtype=torch.int64)[:, None]
                            .expand_as(alpha))
                node_alpha = zero_node_state.clone().scatter_add(0, node_idx, alpha)
                cand_graphs.ndata['alpha'] = node_alpha
                cand_graphs.apply_edges(
                    func=lambda edges: {'alpha': edges.src['alpha']},
                )

        cand_line_graph.ndata.update(cand_graphs.edata)
        for i in range(self.depth - 1):
            cand_line_graph.update_all(fn.copy_u('msg', 'msg'), fn.sum(msg='msg', out='accum_msg'))
            cand_line_graph.apply_nodes(self.loopy_bp_updater)

        cand_graphs.edata.update(cand_line_graph.ndata)

        cand_graphs.update_all(fn.copy_e('msg', 'msg'), fn.sum('msg', 'm'))
        if PAPER:
            cand_graphs.update_all(fn.copy_e('alpha', 'alpha'), fn.sum('alpha', 'accum_alpha'))
        cand_graphs.apply_nodes(self.gather_updater)

        return cand_graphs
