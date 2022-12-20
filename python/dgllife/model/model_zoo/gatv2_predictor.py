# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GATv2-based model for regression and classification on graphs
#
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn

from .mlp_predictor import MLPPredictor
from ..gnn.gatv2 import GATv2
from ..readout.weighted_sum_and_max import WeightedSumAndMax

# pylint: disable=W0221
class GATv2Predictor(nn.Module):
    r"""GATv2-based model for regression and classification on graphs

    GATv2 is introduced in `How Attentive Are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`. This model is based on GATv2 and
    can be used for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with
    learnable weights and max pooling on them and concatenate the output of the
    two operations, which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e. values before
    sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features
    hidden_feats : list of int, optional
        ``hidden_feats[i]`` gives the output size of an attention head in the
        i-th GATv2 layer. ``len(hidden_feats)`` equals the number of GATv2
        layers. By default, we use ``[32, 32]``.
    num_heads : list of int, optional
        ``num_heads[i]`` gives the number of attention heads in the i-th GATv2
        layer. ``len(num_heads)`` equals the number of GATv2 layers. By default
        , we use 4 attention heads per GATv2 layer.
    feat_drops : list of float, optional
        ``feat_drops[i]`` gives the dropout applied to the input features in
        the i-th GATv2 layer. ``len(feat_drops)`` equals the number of GATv2
        layers. By default, we use zero for all GATv2 layers.
    attn_drops : list of float, optional
        ``attn_drops[i]`` gives the dropout applied to the attention values of
        edges in the i-th GATv2 layer. ``len(attn_drops)`` equals the number of
        GATv2 layers. By default, we use zero for all GATv2 layers.
    alphas : list of float, optional
        ``alphas[i]`` gives the slope for the negative values in the LeakyReLU
        function of the i-th GATv2 layer. ``len(alphas)`` equals the number of
        GATv2 layers. By default, we use 0.2 for all GATv2 layers.
    residuals : list of bool, optional
        ``residuals[i]`` decides if residual connection is to be used for the
        i-th GATv2 layer. ``len(residuals)`` equals the number of GATv2 layers.
        By default, we use ``False`` for all GATv2 layers.
    activations : list of callable, optional
        ``activations[i]`` gives the activation function applied to the result
        of the i-th GATv2 layer. ``len(activations)`` equals the number of
        GATv2 layers. By default, we use ELU for all GATv2 layers, except for
        the last layer.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes
        will be invalid since no messages will be passed to those nodes. This
        is harmful for some applications, causing silent performance regression
        . This module will raise a DGLError if it detects 0-in-degree nodes in
        input graph. By setting True, it will suppress the check and let the
        users handle it by themselves. Defaults: False.
    biases : list of bool, optional
        ``biases[i]`` decides if an additive bias is allowed to be learned by
        the i-th GATv2 layer. ``len(biases)`` equals the number of GATv2
        layers. By default, additive biases are learned for all GATv2 layers.
    share_weights : list of bool, optional
        ``share_weights[i]`` decides if the learnable weight matrix for source
        and destination nodes is the same in the i-th GATv2 layer.
        ``len(share_weights)`` equals the number of GATv2 Layers.
        By default, no weight sharing is used in all GATv2 layers.
    agg_modes : list of str, optional
        ``agg_modes[i]`` gives the way to aggregate multi-head attention
        results in the i-th GATv2 layer. ``len(agg_modes)`` equals the number
        of GATv2 Layers. By default, we flatten all-head results for each GATv2
        layer, except for the last layer.
    n_tasks : int, optional
        Number of tasks, which is also the output size. Default to 1.
    predictor_out_feats : int, optional
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float, optional
        The probability for dropout in the output MLP predictor. Default to 0.
    """
    def __init__(
        self,
        in_feats,
        hidden_feats=None,
        num_heads=None,
        feat_drops=None,
        attn_drops=None,
        alphas=None,
        residuals=None,
        activations=None,
        allow_zero_in_degree=False,
        biases=None,
        share_weights=None,
        agg_modes=None,
        n_tasks=1,
        predictor_out_feats=128,
        predictor_dropout=0.):
        super(GATv2Predictor, self).__init__()

        self.gnn = GATv2(in_feats=in_feats,
                         hidden_feats=hidden_feats,
                         num_heads=num_heads,
                         feat_drops=feat_drops,
                         attn_drops=attn_drops,
                         alphas=alphas,
                         residuals=residuals,
                         activations=activations,
                         allow_zero_in_degree=allow_zero_in_degree,
                         biases=biases,
                         share_weights=share_weights,
                         agg_modes=agg_modes)

        if agg_modes[-1] == 'flatten':
            gnn_out_feats = hidden_feats[-1] * num_heads[-1]
        else:
            gnn_out_feats = hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_out_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats, get_attention=False):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs.
            * M1 is the input node feature size, which equals in_feats in
              initialization
        get_attention : bool, optional
            Whether to return the attention values. Defaults: False

        Returns
        -------
        preds : FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        attentions : list of FloatTensor of shape (E, H, 1), optional
            It is returned when :attr:`get_attention` is True.
            ``attentions[i]`` gives the attention values in the i-th GATv2
            layer.

            * `E` is the number of edges.
            * `H` is the number of attention heads.
        """
        if get_attention:
            node_feats, attentions = self.gnn(bg, feats, get_attention=get_attention)
            graph_feats = self.readout(bg, node_feats)
            return self.predict(graph_feats), attentions
        else:
            node_feats = self.gnn(bg, feats)
            graph_feats = self.readout(bg, node_feats)
            return self.predict(graph_feats)
