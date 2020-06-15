# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Convolutional Networks

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import dgl.function as fn

__all__ = ['GCN']


class GraphConv(nn.Module):
    r"""Single GCN layer from `Open Graph Benchmark: Datasets for Machine Learning on Graphs
    <https://arxiv.org/abs/2005.00687>`__

    Parameters
    ----------
    feats : int
        Number of input and output node features.
    """
    def __init__(self, feats):
        super(GraphConv, self).__init__()
        self._feats = feats
        self.linear = nn.Linear(feats, feats)
        self.root_emb = nn.Embedding(1, feats)
        self.edge_encoder = nn.Linear(7, feats)

    def forward(self, graph, nfeat, efeat, degs, norm):
        """Update node representations.

        Parameters
        ----------
        graph : DGLGraph
            DGLGraph for a batch of graphs
        nfeats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size
        efeats : FloatTensor of shape (E, M2)
            * E is the total number of edges in the batch of graphs
            * M2 is the input edge feature size
        norm : FloatTensor of shape (E, 1)
            * E is the total number of edges in the batch of graphs
        Returns
        -------
        new_feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the output node feature size which is equal to input node feature size
        """
        nfeat = self.linear(nfeat)
        edge_embedding = self.edge_encoder(efeat)
        graph = graph.local_var()

        graph.ndata['h'] = nfeat
        graph.apply_edges(fn.copy_src('h', 'e'))
        graph.edata['e'] = norm * F.relu(graph.edata['e'] + edge_embedding)
        graph.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'ft'))
        rst = graph.ndata['ft'] + F.relu(nfeat + self.root_emb.weight) * 1. / degs.view(-1, 1)

        return rst


# GCN model
class GCN(nn.Module):
    r"""GCN from `Open Graph Benchmark: Datasets for Machine Learning on Graphs
    <https://arxiv.org/abs/2005.00687>`__

    Parameters
    ----------
    n_hidden : int
        Number of hidden node features.
    n_classes : int
        Number of output node features.
    n_layers : int
        Number of GCN layers.
    dropout : float
        Dropout probability on the output of each layer.
    JK : string
        It could be ``last`` or ``sum`` which are the ways of the composition of graph representation.
    graph_pooling : string
        It could be ``mean``, ``sum`` or ``max`` which are the types of graph pooling.
    Virtual_node : bool
        Whether to add Virtual Nodes,
        <https://arxiv.org/abs/1704.01212>`__
    residual : bool
        Whether to to residual operation of each layer.
    """
    def __init__(self,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 JK="last",
                 graph_pooling="mean",
                 VirtualNode=False,
                 residual=False):
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.JK = JK
        self.VirtualNode = VirtualNode
        if graph_pooling == "mean":
            self.pool = AvgPooling()
        elif graph_pooling == "sum":
            self.pool = SumPooling()
        elif graph_pooling == "max":
            self.pool = MaxPooling()

        # uniform input node embedding
        self.node_encoder = torch.nn.Embedding(1, n_hidden)

        #Virtual Node embedding
        if VirtualNode == True:
            self.virtualnode_embedding = torch.nn.Embedding(1, n_hidden)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()
            for layer in range(n_layers - 1):
                self.mlp_virtualnode_list.append(nn.Sequential(nn.Linear(n_hidden, 2 * n_hidden),
                                                               nn.BatchNorm1d(2 * n_hidden),
                                                               nn.ReLU(),
                                                               nn.Linear(2 * n_hidden, n_hidden),
                                                               nn.BatchNorm1d(n_hidden),
                                                               nn.ReLU()))

        self.layers = nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        # hidden layers
        for i in range(n_layers):
            self.layers.append(GraphConv(n_hidden))
            self.batch_norms.append(torch.nn.BatchNorm1d(n_hidden))

        # graph readout
        self.graph_pred_linear = torch.nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.residual = residual

    def forward(self, g, degs=None):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        degs : FloatTensor of shape (N, )
            The degrees of each node in the batch of graphs
            * N is the total number of nodes in the batch of graphs

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              the number of classses.
        """
        nfeats = g.ndata['h']
        efeats = g.edata['feat']


        if degs is not None:
            norm = torch.pow(degs, -0.5).unsqueeze(-1)  # (N, 1)
            g.ndata['norm'] = norm
            g.apply_edges(fn.u_mul_v('norm', 'norm', 'norm'))
            norm = g.edata.pop('norm')

        if self.VirtualNode == True:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(g.batch_size).to(nfeats.dtype).to(nfeats.device))
        h_list = [self.node_encoder(nfeats)]
        for layer in range(self.n_layers):
            if self.VirtualNode == True:
                virtualnode_embedding_broadcast = dgl.broadcast_nodes(g, virtualnode_embedding)
                h_list[layer] = h_list[layer] + virtualnode_embedding_broadcast

            if degs is None:
                h = self.layers[layer](g, h_list[layer], efeats)
            else:
                h = self.layers[layer](g, h_list[layer], efeats, degs, norm)
            h = self.batch_norms[layer](h)

            if layer != self.n_layers - 1:
                h = F.relu(h)
            h = self.dropout(h)
            if self.residual:
                h = h + h_list[layer]
            h_list.append(h)

            if self.VirtualNode == True and layer < self.n_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = self.pool2(g, h_list[layer]) + virtualnode_embedding
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + self.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp))
                else:
                    virtualnode_embedding = self.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp))


        # node_representation
        node_representation = 0
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.n_layers):
                node_representation += h_list[layer]

        # graph_readout
        hg = self.pool(g, node_representation)
        hg = self.graph_pred_linear(hg)

        return hg