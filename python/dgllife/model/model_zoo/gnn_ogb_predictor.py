# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Variant of Graph Convolutional Networks/Graph Isomorphism Networks
# for graph property prediction in OGB's Examples

import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from ..gnn.gnn_ogb import GNNOGB

__all__ = ['GNNOGBPredictor']

# pylint: disable=W0221, C0103
class GNNOGBPredictor(nn.Module):
    r"""Variant of GCN/GIN from `Open Graph Benchmark: Datasets for Machine Learning on Graphs
    <https://arxiv.org/abs/2005.00687>`__ for graph property prediction

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
    n_tasks : int
        Number of output tasks. (Default: 1)
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
    readout : str
        The readout function for computing graph-level representations out of node
        representations, which can be 'mean', 'sum' or 'max'. (Default: 'mean')
    """
    def __init__(self,
                 in_edge_feats,
                 num_node_types=1,
                 hidden_feats=300,
                 n_layers=5,
                 n_tasks=1,
                 batchnorm=True,
                 activation=F.relu,
                 dropout=0.,
                 gnn_type='gcn',
                 virtual_node=True,
                 residual=False,
                 jk=False,
                 readout='mean'):
        super(GNNOGBPredictor, self).__init__()

        assert gnn_type in ['gcn', 'gin'], \
            "Expect gnn_type to be 'gcn' or 'gin', got {}".format(gnn_type)
        assert readout in ['mean', 'sum', 'max'], \
            "Expect readout to be in ['mean', 'sum', 'max'], got {}".format(readout)

        self.gnn = GNNOGB(in_edge_feats=in_edge_feats,
                          num_node_types=num_node_types,
                          hidden_feats=hidden_feats,
                          n_layers=n_layers,
                          batchnorm=batchnorm,
                          activation=activation,
                          dropout=dropout,
                          gnn_type=gnn_type,
                          virtual_node=virtual_node,
                          residual=residual,
                          jk=jk)

        if readout == 'mean':
            self.readout = AvgPooling()
        if readout == 'sum':
            self.readout = SumPooling()
        if readout == 'max':
            self.readout = MaxPooling()

        self.predict = nn.Linear(hidden_feats, n_tasks)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gnn.reset_parameters()
        self.predict.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Predict graph properties.

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
        FloatTensor of shape (B, n_tasks)
            Predicted graph properties
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)

        return self.predict(graph_feats)
