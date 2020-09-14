# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=C0111, C0103, E1101, W0611, W0612, I1101, W0221
# pylint: disable=redefined-outer-name

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl import mean_nodes

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6

mpn_loopy_bp_msg = fn.copy_src(src='msg', out='msg')
mpn_loopy_bp_reduce = fn.sum(msg='msg', out='accum_msg')

class LoopyBPUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(LoopyBPUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.W_h.reset_parameters()

    def forward(self, nodes):
        msg_input = nodes.data['msg_input']
        msg_delta = self.W_h(nodes.data['accum_msg'])
        msg = F.relu(msg_input + msg_delta)
        return {'msg': msg}

mpn_gather_msg = fn.copy_edge(edge='msg', out='msg')
mpn_gather_reduce = fn.sum(msg='msg', out='m')

class GatherUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(GatherUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.W_o.reset_parameters()

    def forward(self, nodes):
        m = nodes.data['m']
        return {
            'h': F.relu(self.W_o(torch.cat([nodes.data['x'], m], 1))),
        }

class DGLMPN(nn.Module):
    def __init__(self, hidden_size, depth):
        super(DGLMPN, self).__init__()

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

    def forward(self, mol_graph):
        mol_line_graph = dgl.line_graph(mol_graph, backtracking=False, shared=True)
        mol_line_graph._node_frames = mol_graph._edge_frames

        mol_graph = self.run(mol_graph, mol_line_graph)

        # TODO: replace with unbatch or readout
        g_repr = mean_nodes(mol_graph, 'h')

        return g_repr

    def run(self, mol_graph, mol_line_graph):
        n_nodes = mol_graph.num_nodes()

        mol_graph.apply_edges(
            func=lambda edges: {'src_x': edges.src['x']},
        )

        e_repr = mol_line_graph.ndata
        bond_features = e_repr['x']
        source_features = e_repr['src_x']

        features = torch.cat([source_features, bond_features], 1)
        msg_input = self.W_i(features)
        mol_line_graph.ndata.update({
            'msg_input': msg_input,
            'msg': F.relu(msg_input),
            'accum_msg': torch.zeros_like(msg_input),
        })
        mol_graph.ndata.update({
            'm': bond_features.new(n_nodes, self.hidden_size).zero_(),
            'h': bond_features.new(n_nodes, self.hidden_size).zero_(),
        })

        for i in range(self.depth - 1):
            mol_line_graph.update_all(
                mpn_loopy_bp_msg,
                mpn_loopy_bp_reduce,
                self.loopy_bp_updater,
            )

        mol_graph.edata.update({
            'msg': mol_line_graph.ndata['msg']
        })
        mol_graph.update_all(
            mpn_gather_msg,
            mpn_gather_reduce,
            self.gather_updater,
        )

        return mol_graph
