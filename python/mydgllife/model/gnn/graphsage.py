# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GraphSAGE
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import SAGEConv

__all__ = ['GraphSAGE']

# pylint: disable=W0221, C0103
class GraphSAGE(nn.Module):
    r"""GraphSAGE from `Inductive Representation Learning on Large Graphs
    <https://arxiv.org/abs/1706.02216>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GraphSAGE layer.
        ``len(hidden_feats)`` equals the number of GraphSAGE layers.  By default, we use
        ``[64, 64]``.
    activation : list of activation functions or None
        If not None, ``activation[i]`` gives the activation function to be used for
        the i-th GraphSAGE layer. ``len(activation)`` equals the number of GraphSAGE layers.
        By default, ReLU is applied for all GraphSAGE layers.
    dropout : list of float or None
        ``dropout[i]`` decides the dropout probability on the output of the i-th GraphSAGE layer.
        ``len(dropout)`` equals the number of GraphSAGE layers. By default, no dropout is
        performed for all layers.
    aggregator_type : list of str
        ``aggregator_type[i]`` decides the aggregator type for the i-th GraphSAGE layer, which
        can be one of ``'mean'``, ``'gcn'``, ``'pool'``, ``'lstm'``. By default, we use
        ``'mean'`` for all layers.
    """
    def __init__(self,
                 in_feats,
                 hidden_feats=None,
                 activation=None,
                 dropout=None,
                 aggregator_type=None):
        super(GraphSAGE, self).__init__()

        if hidden_feats is None:
            hidden_feats = [64, 64]

        n_layers = len(hidden_feats)
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        if aggregator_type is None:
            aggregator_type = ['mean' for _ in range(n_layers)]
        lengths = [len(hidden_feats), len(activation), len(dropout), len(aggregator_type)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, activation, ' \
                                       'dropout and aggregator_type to be the same, ' \
                                       'got {}'.format(lengths)

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(SAGEConv(in_feats, hidden_feats[i], aggregator_type[i],
                                            dropout[i], activation[i]))
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
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
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats
