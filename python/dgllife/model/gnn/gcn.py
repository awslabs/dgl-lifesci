# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Convolutional Networks
# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


__all__ = ['GCN']

class GraphConvOgbppa(nn.Module):
    r"""Single GCN layer from `Open Graph Benchmark: Datasets for Machine Learning on Graphs
    <https://arxiv.org/abs/2005.00687>`__

    Parameters
    ----------
    in_feats : int
        Number of input and output node features.
    out_feats : int
        Number of input and output node features.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self, in_feats, out_feats, activation=None):
        super(GraphConvOgbppa, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.root_emb = nn.Embedding(1, out_feats)
        self.edge_encoder = nn.Linear(7, out_feats)
        self._activation = activation

    def forward(self, graph, nfeat, efeat, degs, norm):
        """Update node representations.

        Parameters
        ----------
        graph : DGLGraph
            DGLGraph for a batch of graphs
        nfeat : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size
        efeat : FloatTensor of shape (E, M2)
            * E is the total number of edges in the batch of graphs
            * M2 is the input edge feature size
        degs : FloatTensor of shape (N, 1)
            Degree of each node in the batch of graphs
            * N is the total number of nodes in the batch of graphs
        norm : FloatTensor of shape (E, 1)
            The normalizer of each edge in the batch of graphs
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

        if self._activation is not None:
            rst = self._activation(rst)
        return rst

# pylint: disable=W0221, C0103
class GCNLayer(nn.Module):
    r""" GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__
    or from `Open Graph Benchmark: Datasets for Machine Learning on Graphs
    <https://arxiv.org/abs/2005.00687>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    out_feats : int
        Number of output node features.
    activation : activation function
        Default to be None.
    residual : bool
        Whether to use residual connection, default to be True.
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    gcn_type : string
        The type of GCN. Default to be None,
        or to be 'ogbg-ppa' to use GCNlayer defined in specified task.
    """
    def __init__(self, in_feats, out_feats, activation=None,
                 residual=True, batchnorm=True, dropout=0., gcn_type=None):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.gcn_type = gcn_type
        if gcn_type == None:
            self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                        norm='none', activation=activation)
        elif gcn_type == 'ogbg-ppa':
            self.graph_conv = GraphConvOgbppa(in_feats=in_feats, out_feats=out_feats,
                                              activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, g, feats, efeats=None, degs=None, norm=None):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match in_feats in initialization
        efeats : FloatTensor of shape (E, M2)
            * E is the total number of edges in the batch of graphs
            * M2 is the input edge feature size
        degs : FloatTensor of shape (N, 1)
            Degree of each node in the batch of graphs
            * N is the total number of nodes in the batch of graphs
        norm : FloatTensor of shape (E, 1)
            The normalizer of each edge in the batch of graphs
            * E is the total number of edges in the batch of graphs

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output node feature size, which must match out_feats in initialization
        """
        if self.gcn_type == None:
            new_feats = self.graph_conv(g, feats)
        elif self.gcn_type == 'ogbg-ppa':
            new_feats = self.graph_conv(g, feats, efeats, degs, norm)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats

class GCN(nn.Module):
    r"""GCN from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__
    or from `Open Graph Benchmark: Datasets for Machine Learning on Graphs
    <https://arxiv.org/abs/2005.00687>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers.  By default, we use
        ``[64, 64]``.
    activation : list of activation functions or None
        If None, no activation will be applied. If not None, ``activation[i]`` gives the
        activation function to be used for the i-th GCN layer. ``len(activation)`` equals
        the number of GCN layers. By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    gcn_type : string
        The type of GCN. Default to be None,
        or to be 'ogbg-ppa' to use GCNlayer defined in specified task.
    Virtual_node : bool
        Whether to add Virtual Nodes,
        <https://arxiv.org/abs/1704.01212>`__
    JK : string
        It could be ``last`` or ``sum`` which are the ways of the composition of graph representation.

    """
    def __init__(self, in_feats, hidden_feats=None, activation=None, residual=None,
                 batchnorm=None, dropout=None, gcn_type=None, VirtualNode=False, JK="last",):
        super(GCN, self).__init__()
        self.gcn_type = gcn_type
        self.JK = JK
        self.VirtualNode = VirtualNode

        if hidden_feats is None:
            hidden_feats = [64, 64]

        n_layers = len(hidden_feats)
        self.n_layers = n_layers
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        lengths = [len(hidden_feats), len(activation),
                   len(residual), len(batchnorm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, activation, ' \
                                       'residual, batchnorm and dropout to be the same, ' \
                                       'got {}'.format(lengths)

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GCNLayer(in_feats, hidden_feats[i], activation[i],
                                            residual[i], batchnorm[i], dropout[i], gcn_type))
            in_feats = hidden_feats[i]

        self.pool2 = SumPooling()
        self.dropout = nn.Dropout(p=dropout[0])
        self.residual = residual[0]
        self.node_encoder = torch.nn.Embedding(1, hidden_feats[0])

        if VirtualNode == True:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_feats[0])
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()
            for layer in range(n_layers - 1):
                self.mlp_virtualnode_list.append(nn.Sequential(nn.Linear(hidden_feats[layer], 2 * hidden_feats[layer]),
                                                               nn.BatchNorm1d(2 * hidden_feats[layer]),
                                                               nn.ReLU(),
                                                               nn.Linear(2 * hidden_feats[layer], hidden_feats[layer]),
                                                               nn.BatchNorm1d(hidden_feats[layer]),
                                                               nn.ReLU()))

    def forward(self, g, feats, degs=None):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] in initialization.
        """
        if degs is not None:
            norm = torch.pow(degs, -0.5).unsqueeze(-1)  # (N, 1)
            g.ndata['norm'] = norm
            g.apply_edges(fn.u_mul_v('norm', 'norm', 'norm'))
            norm = g.edata.pop('norm')

        if self.gcn_type == 'ogbg-ppa':
            efeats = g.edata['feat']
            h_list = [self.node_encoder(feats)]

            for layer in range(self.n_layers):
                if self.VirtualNode == True:
                    virtualnode_embedding_broadcast = dgl.broadcast_nodes(g, virtualnode_embedding)
                    h_list[layer] = h_list[layer] + virtualnode_embedding_broadcast

                if degs is not None:
                    h = self.gnn_layers[layer](g, h_list[layer], efeats, degs, norm)
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

            node_representation = 0
            if self.JK == "last":
                node_representation = h_list[-1]
            elif self.JK == "sum":
                node_representation = 0
                for layer in range(self.n_layers):
                    node_representation += h_list[layer]
            return node_representation

        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats
