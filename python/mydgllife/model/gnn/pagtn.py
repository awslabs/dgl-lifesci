# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Path-Augmented Graph Transformer Network
# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax
import dgl.function as fn

__all__ = ['PAGTNGNN']

class PAGTNLayer(nn.Module):
    """
    Single PAGTN layer from `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__

    This will be used for incorporating the information of edge features
    into node features for message passing.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node features.
    edge_feats : int
        Size for the input edge features.
    dropout : float
        The probability for performing dropout. Default to 0.1
    activation : callable
        Activation function to apply. Default to LeakyReLU.
    """
    def __init__(self,
                 node_in_feats,
                 node_out_feats,
                 edge_feats,
                 dropout=0.1,
                 activation=nn.LeakyReLU(0.2)):
        super(PAGTNLayer, self).__init__()
        self.attn_src = nn.Linear(node_in_feats, node_in_feats)
        self.attn_dst = nn.Linear(node_in_feats, node_in_feats)
        self.attn_edg = nn.Linear(edge_feats, node_in_feats)
        self.attn_dot = nn.Linear(node_in_feats, 1)
        self.msg_src = nn.Linear(node_in_feats, node_out_feats)
        self.msg_dst = nn.Linear(node_in_feats, node_out_feats)
        self.msg_edg = nn.Linear(edge_feats, node_out_feats)
        self.wgt_n = nn.Linear(node_in_feats, node_out_feats)
        self.dropout = nn.Dropout(dropout)
        self.act = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_src.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_edg.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_dot.weight, gain=gain)
        nn.init.xavier_normal_(self.msg_src.weight, gain=gain)
        nn.init.xavier_normal_(self.msg_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.msg_edg.weight, gain=gain)
        nn.init.xavier_normal_(self.wgt_n.weight, gain=gain)

    def forward(self, g, node_feats, edge_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats) or (V, n_head, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.

        Returns
        -------
        float32 tensor of shape (V, node_out_feats) or (V, n_head, node_out_feats)
            Updated node features.
        """

        g = g.local_var()
        # In the paper node_src, node_dst, edge feats are concatenated
        # and multiplied with the matrix. We have optimized this step
        # by having three separate matrix multiplication.
        g.ndata['src'] = self.dropout(self.attn_src(node_feats))
        g.ndata['dst'] = self.dropout(self.attn_dst(node_feats))
        edg_atn = self.dropout(self.attn_edg(edge_feats)).unsqueeze(-2)
        g.apply_edges(fn.u_add_v('src', 'dst', 'e'))
        atn_scores = self.act(g.edata.pop('e') + edg_atn)

        atn_scores = self.attn_dot(atn_scores)
        atn_scores = self.dropout(edge_softmax(g, atn_scores))

        g.ndata['src'] = self.msg_src(node_feats)
        g.ndata['dst'] = self.msg_dst(node_feats)
        g.apply_edges(fn.u_add_v('src', 'dst', 'e'))
        atn_inp = g.edata.pop('e') + self.msg_edg(edge_feats).unsqueeze(-2)
        atn_inp = self.act(atn_inp)
        g.edata['msg'] = atn_scores * atn_inp
        g.update_all(fn.copy_e('msg', 'm'), fn.sum('m', 'feat'))
        out = g.ndata.pop('feat') + self.wgt_n(node_feats)
        return self.act(out)


class PAGTNGNN(nn.Module):
    """Multilayer PAGTN model for updating node representations.
    PAGTN is introduced in `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node features.
    node_hid_feats : int
        Size for the hidden node features.
    edge_feats : int
        Size for the input edge features.
    depth : int
        Number of PAGTN layers to be applied.
    nheads : int
        Number of attention heads.
    dropout : float
        The probability for performing dropout. Default to 0.1
    activation : callable
        Activation function to apply. Default to LeakyReLU.
    """

    def __init__(self,
                 node_in_feats,
                 node_out_feats,
                 node_hid_feats,
                 edge_feats,
                 depth=5,
                 nheads=1,
                 dropout=0.1,
                 activation=nn.LeakyReLU(0.2)):
        super(PAGTNGNN, self).__init__()
        self.depth = depth
        self.nheads = nheads
        self.node_hid_feats = node_hid_feats
        self.atom_inp = nn.Linear(node_in_feats, node_hid_feats * nheads)
        self.model = nn.ModuleList([PAGTNLayer(node_hid_feats, node_hid_feats,
                                               edge_feats, dropout,
                                               activation)
                                    for _ in range(depth)])
        self.atom_out = nn.Linear(node_in_feats + node_hid_feats * nheads, node_out_feats)
        self.act = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.atom_inp.weight, gain=gain)
        nn.init.xavier_normal_(self.atom_out.weight, gain=gain)
        self.model.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.

        Returns
        -------
        float32 tensor of shape (V, node_out_feats)
            Updated node features.
        """
        g = g.local_var()
        atom_input = self.atom_inp(node_feats).view(-1, self.nheads, self.node_hid_feats)
        atom_input = self.act(atom_input)
        atom_h = atom_input
        for i in range(self.depth):
            attn_h = self.model[i](g, atom_h, edge_feats)
            atom_h = torch.nn.functional.relu(attn_h + atom_input)

        atom_h = atom_h.view(-1, self.nheads*self.node_hid_feats)
        atom_output = torch.cat([node_feats, atom_h], dim=1)
        atom_h = torch.nn.functional.relu(self.atom_out(atom_output))
        return atom_h
