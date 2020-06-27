# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Variant of Graph Convolutional Networks/Graph Isomorphism Networks in OGB's Examples

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.glob import SumPooling

__all__ = ['GNNOGB']

# pylint: disable=W0221, C0103
class GCNOGBLayer(nn.Module):
    r"""Variant of Single GCN layer from `Open Graph Benchmark: Datasets for Machine Learning on
    Graphs <https://arxiv.org/abs/2005.00687>`__

    Parameters
    ----------
    in_node_feats : int
        Number of input node features.
    in_edge_feats : int
        Number of input edge features.
    out_feats : int
        Number of output node features.
    """
    def __init__(self, in_node_feats, in_edge_feats, out_feats):
        super(GCNOGBLayer, self).__init__()

        self.project_in_node_feats = nn.Linear(in_node_feats, out_feats)
        self.project_in_edge_feats = nn.Linear(in_edge_feats, out_feats)
        self.project_residual = nn.Embedding(1, out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_in_node_feats.reset_parameters()
        self.project_in_edge_feats.reset_parameters()
        self.project_residual.reset_parameters()

    def forward(self, g, node_feats, edge_feats, degs, norm):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : FloatTensor of shape (N, in_node_feats)
            Input node features. N for the total number of nodes in the batch of graphs.
        edge_feats : FloatTensor of shape (E, in_edge_feats)
            Input edge features. E for the total number of edges in the batch of graphs.
        degs : FloatTensor of shape (N, 1)
            Degree of each node in the batch of graphs.
            N for the total number of nodes in the batch of graphs.
        norm : FloatTensor of shape (E, 1)
            Edge-associated normalization coefficients. For an edge (i, j), this is
            equivalent to 1 / sqrt{d_i * d_j}, d_i and d_j are separately the degree
            of node i and j.

        Returns
        -------
        FloatTensor of shape (N, out_feats)
            Updated node representations.
        """
        g = g.local_var()
        node_feats = self.project_in_node_feats(node_feats)
        edge_feats = self.project_in_edge_feats(edge_feats)

        g.ndata['feat'] = node_feats
        g.apply_edges(fn.copy_u('feat', 'e'))
        edge_feats = F.relu(g.edata['e'] + edge_feats)
        g.edata['e'] = norm * edge_feats
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'feat'))

        residual_node_feats = node_feats + self.project_residual.weight
        residual_node_feats = F.relu(residual_node_feats)
        residual_node_feats = residual_node_feats * 1. / degs.view(-1, 1)

        rst = g.ndata['feat'] + residual_node_feats

        return rst

class GINOGBLayer(nn.Module):
    r"""Variant of Single GIN layer from `Open Graph Benchmark: Datasets for Machine Learning on
    Graphs <https://arxiv.org/abs/2005.00687>`__

    Parameters
    ----------
    node_feats : int
        Number of input and output node features.
    in_edge_feats : int
        Number of input edge features.
    """
    def __init__(self, node_feats, in_edge_feats):
        super(GINOGBLayer, self).__init__()

        self.project_in_edge_feats = nn.Linear(in_edge_feats, node_feats)
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.project_out = nn.Sequential(
            nn.Linear(node_feats, 2 * node_feats),
            nn.BatchNorm1d(2 * node_feats),
            nn.ReLU(),
            nn.Linear(2 * node_feats, node_feats)
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_in_edge_feats.reset_parameters()
        device = self.eps.device
        self.eps = nn.Parameter(torch.Tensor([0]).to(device))
        for layer in self.project_out:
            if isinstance(layer, (nn.Linear, nn.BatchNorm1d)):
                layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : FloatTensor of shape (N, node_feats)
            Input node features. N for the total number of nodes in the batch of graphs.
        edge_feats : FloatTensor of shape (E, in_edge_feats)
            Input edge features. E for the total number of edges in the batch of graphs.

        Returns
        -------
        FloatTensor of shape (N, node_feats)
            Updated node representations.
        """
        g = g.local_var()
        edge_feats = self.project_in_edge_feats(edge_feats)

        g.ndata['feat'] = node_feats
        g.apply_edges(fn.copy_u('feat', 'e'))
        g.edata['e'] = F.relu(edge_feats + g.edata['e'])
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'feat'))

        rst = g.ndata['feat']
        rst = self.project_out(rst + (1 + self.eps) * node_feats)

        return rst

class GNNOGB(nn.Module):
    r"""Variant of GCN/GIN from `Open Graph Benchmark: Datasets for Machine Learning on Graphs
    <https://arxiv.org/abs/2005.00687>`__

    Parameters
    ----------
    in_edge_feats : int
        Number of input edge features.
    num_node_types : int
        Number of node types to embed. (Default: 1)
    hidden_feats : int
        Size for hidden representations. (Default: 300)
    n_layers : int
        Number of GNN layers to use. (Default: 5)
    batchnorm : bool
        Whether to apply batch normalization. (Default: True)
    activation : callable or None
        Activation function to apply to the output of each GNN layer except for the last layer.
        If None, no activation will be applied. (Default: ReLU)
    dropout : float
        The probability for dropout. (Default: 0, i.e. no dropout)
    gnn_type : str
        The GNN type to use, which can be either 'gcn' or 'gin'. (Default: 'gcn')
    virtual_node : bool
        Whether to use virtual node. (Default: True)
    residual : bool
        Whether to apply residual connections for virtual node embeddings. (Default: False)
    jk : bool
        Whether to sum over the output of all GNN layers as in
        `JK networks <https://arxiv.org/abs/1806.03536>`__. (Default: False)
    """
    def __init__(self,
                 in_edge_feats,
                 num_node_types=1,
                 hidden_feats=300,
                 n_layers=5,
                 batchnorm=True,
                 activation=F.relu,
                 dropout=0.,
                 gnn_type='gcn',
                 virtual_node=True,
                 residual=False,
                 jk=False):
        super(GNNOGB, self).__init__()

        assert gnn_type in ['gcn', 'gin'], \
            "Expect gnn_type to be either 'gcn' or 'gin', got {}".format(gnn_type)

        self.n_layers = n_layers
        # Initial node embeddings
        self.node_encoder = nn.Embedding(num_node_types, hidden_feats)
        # Hidden layers
        self.layers = nn.ModuleList()
        self.gnn_type = gnn_type
        for _ in range(n_layers):
            if gnn_type == 'gcn':
                self.layers.append(GCNOGBLayer(in_node_feats=hidden_feats,
                                               in_edge_feats=in_edge_feats,
                                               out_feats=hidden_feats))
            else:
                self.layers.append(GINOGBLayer(node_feats=hidden_feats,
                                               in_edge_feats=in_edge_feats))

        self.virtual_node = virtual_node
        if virtual_node:
            self.virtual_node_emb = nn.Embedding(1, hidden_feats)
            self.mlp_virtual_project = nn.ModuleList()
            for _ in range(n_layers - 1):
                self.mlp_virtual_project.append(nn.Sequential(
                    nn.Linear(hidden_feats, 2 * hidden_feats),
                    nn.BatchNorm1d(2 * hidden_feats),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_feats, hidden_feats),
                    nn.BatchNorm1d(hidden_feats),
                    nn.ReLU()))
            self.virtual_readout = SumPooling()

        if batchnorm:
            self.batchnorms = nn.ModuleList()
            for _ in range(n_layers):
                self.batchnorms.append(nn.BatchNorm1d(hidden_feats))
        else:
            self.batchnorms = None

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.jk = jk

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.node_encoder.reset_parameters()
        for gnn_layer in self.layers:
            gnn_layer.reset_parameters()

        if self.virtual_node:
            nn.init.constant_(self.virtual_node_emb.weight.data, 0)
            for mlp_layer in self.mlp_virtual_project:
                for layer in mlp_layer:
                    if isinstance(layer, (nn.Linear, nn.BatchNorm1d)):
                        layer.reset_parameters()

        if self.batchnorms is not None:
            for norm_layer in self.batchnorms:
                norm_layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : LongTensor of shape (N, 1)
            Input categorical node features. N for the number of nodes.
        edge_feats : FloatTensor of shape (E, in_edge_feats)
            Input edge features. E for the number of edges.

        Returns
        -------
        FloatTensor of shape (N, hidden_feats)
            Output node representations
        """
        if self.gnn_type == 'gcn':
            degs = (g.in_degrees().float() + 1).to(node_feats.device)
            norm = torch.pow(degs, -0.5).unsqueeze(-1)                # (N, 1)
            g.ndata['norm'] = norm
            g.apply_edges(fn.u_mul_v('norm', 'norm', 'norm'))
            norm = g.edata.pop('norm')

        if self.virtual_node:
            virtual_node_feats = self.virtual_node_emb(
                torch.zeros(g.batch_size).to(node_feats.dtype).to(node_feats.device))
        h_list = [self.node_encoder(node_feats)]

        for l in range(len(self.layers)):
            if self.virtual_node:
                virtual_feats_broadcast = dgl.broadcast_nodes(g, virtual_node_feats)
                h_list[l] = h_list[l] + virtual_feats_broadcast

            if self.gnn_type == 'gcn':
                h = self.layers[l](g, h_list[l], edge_feats, degs, norm)
            else:
                h = self.layers[l](g, h_list[l], edge_feats)

            if self.batchnorms is not None:
                h = self.batchnorms[l](h)

            if self.activation is not None and l != self.n_layers - 1:
                h = self.activation(h)
            h = self.dropout(h)
            h_list.append(h)

            if l < self.n_layers - 1 and self.virtual_node:
                ### Update virtual node representation from real node representations
                virtual_node_feats_tmp = self.virtual_readout(g, h_list[l]) + virtual_node_feats
                if self.residual:
                    virtual_node_feats = virtual_node_feats + self.dropout(
                        self.mlp_virtual_project[l](virtual_node_feats_tmp))
                else:
                    virtual_node_feats = self.dropout(
                        self.mlp_virtual_project[l](virtual_node_feats_tmp))

        if self.jk:
            return torch.stack(h_list, dim=0).sum(0)
        else:
            return h_list[-1]
