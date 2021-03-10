# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Path-Augmented Graph Transformer Network
# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn
from ..gnn import PAGTNGNN
from ..readout import MLPNodeReadout


class PAGTNPredictor(nn.Module):
    """PAGTN model for regression and classification on graphs.

    PAGTN is introduced in `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node features in PAGTN layers.
    node_hid_feats : int
        Size for the hidden node features in PAGTN layers.
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
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    mode : 'max' or 'mean' or 'sum'
        Whether to compute elementwise maximum, mean or sum of the node representations.

    """
    def __init__(self,
                 node_in_feats,
                 node_out_feats,
                 node_hid_feats,
                 edge_feats,
                 depth=5,
                 nheads=1,
                 dropout=0.1,
                 activation=nn.LeakyReLU(0.2),
                 n_tasks=1,
                 mode='sum'):
        super(PAGTNPredictor, self).__init__()
        self.model = PAGTNGNN(node_in_feats, node_out_feats,
                              node_hid_feats, edge_feats,
                              depth, nheads, dropout, activation)
        self.readout = MLPNodeReadout(node_out_feats + node_in_feats,
                                      node_out_feats,
                                      n_tasks,
                                      mode=mode)

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.

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

        atom_h = self.model(g, node_feats, edge_feats)
        atom_h = torch.cat([atom_h, node_feats], dim=1)
        return self.readout(g, atom_h)
