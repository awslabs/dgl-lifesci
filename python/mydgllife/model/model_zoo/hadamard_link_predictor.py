# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Predictor for link prediction by taking elementwise multiplication of node representations

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['HadamardLinkPredictor']

class HadamardLinkPredictor(nn.Module):
    """Link prediction by taking the elementwise multiplication of two node representations

    The elementwise multiplication is also called Hadamard product.

    Parameters
    ----------
    in_feats : int
        Number of input node features
    hidden_feats : int
        Number of hidden features. Default to 256.
    num_layers : int
        Number of linear layers used in total, which should be
        at least 2, counting the input and output layers. Default to 3.
    n_tasks : int
        Number of output tasks. Default to 1.
    dropout : float
        Dropout before each linear layer except for the first one.
        Default to 0., i.e. no dropout is performed.
    activation : callable
        Activation function to apply after the output of each linear layer.
        Default to ReLU.
    """
    def __init__(self,
                 in_feats,
                 hidden_feats=256,
                 num_layers=3,
                 n_tasks=1,
                 dropout=0.,
                 activation=F.relu):
        super(HadamardLinkPredictor, self).__init__()

        assert num_layers >= 2, 'Expect num_layers to be at least 2, got {:d}'.format(num_layers)

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hidden_feats))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_feats, hidden_feats))
        # output layer
        self.layers.append(nn.Linear(hidden_feats, n_tasks))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        # Reset the parameters of the Linear layers
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, left_node_feats, right_node_feats):
        """Link Prediction

        Perform link prediction for P pairs of nodes. Note
        that this model is symmetric and we don't have
        separate parameters for the two arguments.

        Parameters
        ----------
        left_node_feats : float32 tensor of shape (P, D1)
            Representations for the first node in P pairs.
            D1 for the number of input node features.
        right_node_feats : float32 tensor of shape (P, D1)
            Representations for the second node in P pairs.
            D1 for the number of input node features.

        Returns
        -------
        float32 tensor of shape (P, D2)
            Pre-softmax/sigmoid logits, D2 equals n_tasks.
        """
        pair_feats = left_node_feats * right_node_feats
        for layer in self.layers[:-1]:
            pair_feats = layer(pair_feats)
            if self.activation is not None:
                pair_feats = self.activation(pair_feats)
            pair_feats = self.dropout(pair_feats)
        out = self.layers[-1](pair_feats)

        return out
