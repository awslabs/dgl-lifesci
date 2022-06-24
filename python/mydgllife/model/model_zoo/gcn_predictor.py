# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GCN-based model for regression and classification on graphs.
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn

from .mlp_predictor import MLPPredictor
from ..gnn.gcn import GCN
from ..readout.weighted_sum_and_max import WeightedSumAndMax

# pylint: disable=W0221
class GCNPredictor(nn.Module):
    """GCN-based model for regression and classification on graphs.

    GCN is introduced in `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__. This model is based on GCN and can be used
    for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers. By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
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
    classifier_hidden_feats : int
        (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
        in the classifier. Default to 128.
    classifier_dropout : float
        (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
        Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(GCNPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return self.predict(graph_feats)
