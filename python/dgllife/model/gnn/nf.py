# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Molecular Fingerprint
# pylint: disable= no-member, arguments-differ, invalid-name

import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import ModuleList

__all__ = ['NFGNN']

class NFLayer(nn.Module):
    r"""Single convolutional layer from `Convolutional Networks on Graphs for Learning Molecular
    Fingerprints <https://arxiv.org/abs/1509.09292>`__

    NF stands for neural fingerprint.

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    out_feats : int
        Number of output node features.
    max_degree : int, optional
        The maximum node degree to consider when updating weights. Default to be 10.
    activation : activation function, optional
        Default to be None.
    batchnorm : bool, optional
        Whether to apply batch normalization to the output. Default to be True.
    dropout : float, optional
        The probability of dropout for the output. Default to be 0.
    """
    def __init__(self, in_feats, out_feats, max_degree=10, activation=None, batchnorm=True,
                 dropout=0.):
        super(NFLayer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.activation = activation
        self.max_degree = max_degree
        self.lin_zero_deg = nn.Linear(in_feats, out_feats)
        self.lins_l = ModuleList([nn.Linear(in_feats, out_feats)
                                  for _ in range(1, max_degree + 1)])
        self.lins_r = ModuleList([nn.Linear(in_feats, out_feats, bias=False)
                                  for _ in range(1, max_degree + 1)])

        if batchnorm:
            self.bn = nn.BatchNorm1d(out_feats)
        else:
            self.bn = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.lin_zero_deg.reset_parameters()
        for lin in self.lins_l:
            lin.reset_parameters()
        for lin in self.lins_r:
            lin.reset_parameters()

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, g_self, feats, deg=None, max_deg=None, deg_membership=None):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        g_self : DGLGraph
            DGLGraph for a batch of graphs with self loops added
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match in_feats in initialization
        deg : LongTensor of shape (N), optional
            In-degrees of the nodes in the graph.
        max_deg : int, optional
            Max value in :attr:`deg`.
        deg_membership : list of LongTensor, optional
            deg_membership[i] gives a 1D LongTensor for the IDs of nodes with in-degree i.

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output node feature size, which must match out_feats in initialization
        deg : LongTensor of shape (N)
            In-degrees of the nodes in the graph.
        max_deg : int
            Max value in :attr:`deg`.
        deg_membership : list of LongTensor
            deg_membership[i] gives a 1D LongTensor for the IDs of nodes with in-degree i.
        """
        if deg is None:
            deg = g.in_degrees().to(feats.device)
            deg = deg.clamp(max=self.max_degree)
            assert max_deg is None, 'Expect max_deg to be None when deg is None.'

        if max_deg is None:
            max_deg = deg.max().cpu().item()

        if deg_membership is None:
            deg_membership = [
                (deg == i).nonzero(as_tuple=False).view(-1) for i in range(max_deg + 1)
            ]

        with g.local_scope():
            # Message passing
            g.ndata['h'] = feats
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.sum('m', 'h'))
            h = g.ndata.pop('h')

            out = h.new_empty(list(h.size())[:-1] + [self.out_feats])

            # Case for degree 0
            idx = deg_membership[0]
            deg_out = self.lin_zero_deg(feats.index_select(0, idx))
            out.index_copy_(0, idx, deg_out)

            # Degree-based transformation
            for i, (lin_l, lin_r) in enumerate(zip(self.lins_l, self.lins_r)):
                current_deg = i + 1

                if current_deg > max_deg:
                    break

                idx = deg_membership[current_deg]
                deg_out = lin_l(h.index_select(0, idx)) + lin_r(feats.index_select(0, idx))
                out.index_copy_(0, idx, deg_out)

            if self.activation is not None:
                out = self.activation(out)

            if self.bn is not None:
                out = self.bn(out)

            if self.dropout is not None:
                out = self.dropout(out)

        with g_self.local_scope():
            g_self.ndata['h'] = out
            g_self.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.max('m', 'h'))
            out = g_self.ndata['h']

        return out, deg, max_deg, deg_membership

class NFGNN(nn.Module):
    r"""GNN from `Convolutional Networks on Graphs for Learning Molecular
    Fingerprints <https://arxiv.org/abs/1509.09292>`__

    NF stands for neural fingerprint.

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int, optional
        ``hidden_feats[i]`` gives the size of node representations after the i-th NF layer.
        ``len(hidden_feats)`` equals the number of NF layers.  By default, we use
        ``[64, 64]``.
    max_degree : int
        The maximum node degree to consider when updating weights. Default to be 10.
    activation : list of activation functions or None
        If not None, ``activation[i]`` gives the activation function to be used for
        the i-th NF layer. ``len(activation)`` equals the number of NF layers.
        By default, ReLU is applied for all NF layers.
    batchnorm : list of bool, optional
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th NF layer. ``len(batchnorm)`` equals the number of NF layers. By default,
        batch normalization is applied for all NF layers.
    dropout : list of float, optional
        ``dropout[i]`` decides the dropout to be applied on the output of the i-th NF layer.
        ``len(dropout)`` equals the number of NF layers. By default, dropout is not applied
        for all NF layers.
    """
    def __init__(self, in_feats, hidden_feats=None, max_degree=10, activation=None,
                 batchnorm=None, dropout=None):
        super(NFGNN, self).__init__()

        if hidden_feats is None:
            hidden_feats = [64, 64]

        n_layers = len(hidden_feats)
        if activation is None:
            activation = [F.relu] * n_layers

        if batchnorm is None:
            batchnorm = [True] * n_layers

        if dropout is None:
            dropout = [0.] * n_layers

        lengths = [len(hidden_feats), len(activation), len(batchnorm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, activation, ' \
                                       'batchnorm, and dropout to be the same, ' \
                                       'got {}'.format(lengths)

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(NFLayer(in_feats, hidden_feats[i], max_degree, activation[i],
                                           batchnorm[i], dropout[i]))
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
        deg = None
        max_deg = None
        deg_membership = None
        g_self = dgl.add_self_loop(g)
        for gnn in self.gnn_layers:
            feats, deg, max_deg, deg_membership = gnn(g, g_self, feats, deg,
                                                      max_deg, deg_membership)
        return feats
