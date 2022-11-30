# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GATv2-based model for regression and classification on graphs.
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn

from .mlp_predictor import MLPPredictor
from ..gnn.gatv2 import GATv2
from ..readout.weighted_sum_and_max import WeightedSumAndMax

# pylint: disable=W0221
class GATv2Predictor(nn.Module):
    r"""GATv2-based model for regression and classification on graphs.

    GATv2 is introduced in `HOW ATTENTIVE ARE GRAPH ATTENTION NETWORKS? <https://arxiv.org/pdf/2105.14491.pdf>`
    This model is based on GATv2 and can be used for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features
    out_feats : list of int
        ``out_feats[i]`` gives the output size of an attention head in the i-th GATv2 layer.
        ``len(out_feats)`` equals the number of GATv2 layers. By default, we use ``[32, 32]``.
    num_heads : list of int
        ``num_heads[i]`` gives the number of attention heads in the i-th GATv2 layer.
        ``len(num_heads)`` equals the number of GATv2 layers. By default, we use 4 attention heads
        for each GATv2 layer.
    feat_drops : list of float
        ``feat_drops[i]`` gives the dropout applied to the input features in the i-th GATv2 layer.
        ``len(feat_drops)`` equals the number of GATv2 layers. By default, this will be zero for
        all GATv2 layers.
    attn_drops : list of float
        ``attn_drops[i]`` gives the dropout applied to attention values of edges in the i-th GATv2
        layer. ``len(attn_drops)`` equals the number of GATv2 layers. By default, this will be zero
        for all GATv2 layers.
    alphas : list of float
        Hyperparameters in LeakyReLU, which are the slopes for negative values. ``alphas[i]``
        gives the slope for negative value in the i-th GATv2 layer. ``len(alphas)`` equals the
        number of GATv2 layers. By default, this will be 0.2 for all GATv2 layers.
    residuals : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GATv2 layer.
        ``len(residual)`` equals the number of GATv2 layers. By default, residual connection
        is performed for each GATv2 layer.
    agg_modes : list of str
        The way to aggregate multi-head attention results for each GATv2 layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
        GATv2 layer. ``len(agg_modes)`` equals the number of GATv2 layers. By default, we flatten
        multi-head results for intermediate GATv2 layers and compute mean of multi-head results
        for the last GATv2 layer.
    activations : list of activation function or None
        ``activations[i]`` gives the activation function applied to the aggregated multi-head
        results for the i-th GATv2 layer. ``len(activations)`` equals the number of GATv2 layers.
        By default, ELU is applied for intermediate GATv2 layers and no activation is applied
        for the last GATv2 layer.
    biases : list of bool
        ``biases[i]`` gives whether to add bias for the i-th GATv2 layer. ``len(activations)``
        equals the number of GATv2 layers. By default, bias is added for all GATv2 layers.
    share_weights : list of bool, optional
        If weight-sharing is enabled, the same matrix for :math:`W_{left}` and :math:`W_{right}` in
        the above equations, will be applied to the source and the target node of every edge.
        ``share_weights[i]`` decides if weight-sharing is used for the i-th GATv2 Layer.
        ``len(share_weights)`` equals the number of GATv2 Layers.
        By default, no weight-sharing is used for the GATv2 Layers.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_out_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """
    def __init__(self, in_feats, out_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 allow_zero_in_degree=False, share_weights=None,
                 biases=None, n_tasks=1,
                 predictor_out_feats=128, predictor_dropout=0.):
        super(GATv2Predictor, self).__init__()

        self.gnn = GATv2(in_feats=in_feats,
                       out_feats=out_feats,
                       num_heads=num_heads,
                       feat_drops=feat_drops,
                       attn_drops=attn_drops,
                       alphas=alphas,
                       residuals=residuals,
                       agg_modes=agg_modes,
                       activations=activations,
                       biases=biases)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.out_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.out_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_out_feats,
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
