# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Molecular Fingerprint for regression and classification on graphs.
# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn

from ..gnn.nf import NFGNN
from ..readout.sum_and_max import SumAndMax

__all__ = ['NFPredictor']

# pylint: disable=W0221
class NFPredictor(nn.Module):
    """Neural Fingerprint (NF) for regression and classification on graphs.

    NF is introduced in `Convolutional Networks on Graphs for Learning Molecular
    Fingerprints <https://arxiv.org/abs/1509.09292>`__. This model can be used for
    regression and classification on graphs.

    After updating node representations, we perform a sum and max pooling on them
    and concatenate the output of the two operations, which is then fed into an
    MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
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
    predicor_hidden_size : int
        Size for hidden representations in the output MLP predictor. Default to be 128.
    predictor_batchnorm : bool
        Whether to apply batch normalization in the output MLP predictor. Default to be True.
        Default to be True.
    predictor_dropout : float
        The dropout probability in the output MLP predictor. Default to be 0.
    predictor_activation : activation function
        The activation function in the output MLP predictor. Default to be Tanh.
    """
    def __init__(self, in_feats, n_tasks=1, hidden_feats=None, max_degree=10, activation=None,
                 batchnorm=None, dropout=None, predictor_hidden_size=128, predictor_batchnorm=True,
                 predictor_dropout=0., predictor_activation=torch.tanh):
        super(NFPredictor, self).__init__()

        self.gnn = NFGNN(in_feats, hidden_feats, max_degree, activation, batchnorm, dropout)
        gnn_out_feats = self.gnn.gnn_layers[-1].out_feats

        self.node_to_graph = nn.Linear(gnn_out_feats, predictor_hidden_size)
        if predictor_batchnorm:
            self.predictor_bn = nn.BatchNorm1d(predictor_hidden_size)
        else:
            self.predictor_bn = None
        if predictor_dropout > 0:
            self.predictor_dropout = nn.Dropout(predictor_dropout)
        else:
            self.predictor_dropout = None

        self.readout = SumAndMax()
        self.predictor_activation = predictor_activation
        self.predict = nn.Linear(2 * predictor_hidden_size, n_tasks)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gnn.reset_parameters()
        self.node_to_graph.reset_parameters()
        if self.predictor_bn is not None:
            self.predictor_bn.reset_parameters()

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
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        feats = self.gnn(g, feats)
        feats = self.node_to_graph(feats)

        if self.predictor_bn is not None:
            feats = self.predictor_bn(feats)
        if self.predictor_dropout is not None:
            feats = self.predictor_dropout(feats)
        graph_feats = self.readout(g, feats)

        if self.predictor_activation is not None:
            graph_feats = self.predictor_activation(graph_feats)

        return self.predict(graph_feats)
