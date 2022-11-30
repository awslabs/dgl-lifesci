# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Attention Networks
#
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv

__all__ = ["GATv2"]

# pylint: disable=W0221
class GATv2Layer(nn.Module):
    r"""Single GAT layer from `HOW ATTENTIVE ARE GRAPH ATTENTION NETWORKS?
    <https://arxiv.org/pdf/2105.14491.pdf>`

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        If the layer is to be applied to a unidirectional bipartite graph, `in_feats`
        specifies the input feature size on both the source and destination nodes.
        If a scalar is given, the source and destination node feature size
        would take the same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    share_weights : bool, optional
        If set to :obj:`True`, the same matrix for :math:`W_{left}` and :math:`W_{right}` in
        the above equations, will be applied to the source and the target node of every edge.
        (default: :obj:`False`)

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be applied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        share_weights=False,
        agg_mode="flatten",
    ):
        super(GATv2Layer, self).__init__()
        self.gatv2_conv = GATv2Conv(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
            bias=bias,
            share_weights=share_weights,
        )
        assert agg_mode in ["flatten", "mean"]
        self.agg_mode = agg_mode
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gatv2_conv.reset_parameters()

    def forward(self, bg, feats):
        """Update node representations

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              out_feats in initialization if self.agg_mode == 'mean' and
              out_feats * num_heads in initialization otherwise.
        """
        feats = self.gatv2_conv(bg, feats)
        if self.agg_mode == "flatten":
            feats = feats.flatten(1)
        else:
            feats = feats.mean(1)

        if self.activation is not None:
            feats = self.activation(feats)

        return feats


class GATv2(nn.Module):
    r"""GATv2 from `HOW ATTENTIVE ARE GRAPH ATTENTION NETWORKS? <https://arxiv.org/pdf/2105.14491.pdf>`

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        If the layer is to be applied to a unidirectional bipartite graph, `in_feats`
        specifies the input feature size on both the source and destination nodes.
        If a scalar is given, the source and destination node feature size
        would take the same value.
    out_feats : list of int, optional
        ``out_feats[i]`` gives the output size of an attention head in i-th GATv2 Layer
        ``len(out_feats)`` equals the number of GATv2 layers.
        By default, we use ``[32, 32]``
    num_heads : list of int, optional
        ``num_heads[i]`` gives the number of attention heads in the i-th GATv2 Layer.
        ``len(num_heads)`` equals the number of GATv2 Layers.
        By default, we use 4 attention heads per GATv2 Layer.
    feat_drops : list of float, optional
        ``feat_drops[i]`` gives the dropout applied to input features in the i-th GATv2 Layer.
        ``len(feat_drops)`` equals the number of GATv2 Layers.
        By default we use zero for each GATv2 Layer.
    attn_drops : list of float, optional
        ``attn_drops[i]`` gives the dropout applied to the attention values of edges in the i-th GATv2 Layer.
        ``len(attn_drops)`` equals the number of GATv2 Layers.
        By default we use zero for each GATv2 Layer
    alphas : list of float, optional
        ``alphas[i]`` gives the (slope) alpha for the negative values of the (RELU) ELU of the i-th GATv2 Layer.
        ``len(alphas)`` equals the number of Gatv2 Layers
        By default we use ``0.2`` for each GATv2Layer
    residuals : list of bool, optional
        ``residuals[i]`` gives decides if residual connection is to be use for the i-th GATv2 Layer.
        ``len(residuals)`` equals the number of GATv2 Layers
        By default we use ``False`` for each GATv2Layer
    activations : list of callable activation function/layer or None
        ``activations[i]`` gives the activation function applied to the aggregated multi-head results for the i-th GATv2 Layer
        ``len(activations)`` equals the number of GATv2 Layers
        By default, the activation function for each GATv2 Layer is the ELU function, except for the last,
        which has no activation
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    biases : list of bool, optional
        ``biases[i]`` decides if an additive bias is allowed to be learned by the i-th GATv2 Layer.
        ``len(biases)`` equals the number of GATv2 Layers
        By default, additive biases are learned by GATv2 Layers
    share_weights : list of bool, optional
        If weight-sharing is enabled, the same matrix for :math:`W_{left}` and :math:`W_{right}` in
        the above equations, will be applied to the source and the target node of every edge.
        ``share_weights[i]`` decides if weight-sharing is used for the i-th GATv2 Layer.
        ``len(share_weights)`` equals the number of GATv2 Layers.
        By default, no weight-sharing is used for the GATv2 Layers.
    agg_modes : list of str
        The way to aggregate multi-head attention results for each GAT layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th GAT layer.
        ``len(agg_modes)`` equals the number of GAT layers. 
        By default, we flatten all-head results for each GAT layer except the last.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
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
    ):
        super(GATv2, self).__init__()

        if out_feats is None:
            out_feats = [32, 32]

        n_layers = len(out_feats)
        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0.0 for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0.0 for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [True for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ["flatten" for _ in range(n_layers - 1)]
            agg_modes.append("mean")
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        if biases is None:
            biases = [True for _ in range(n_layers)]
        if share_weights is None:
            share_weights = [False for _ in range(n_layers)]
        lengths = [
            len(out_feats),
            len(num_heads),
            len(feat_drops),
            len(attn_drops),
            len(alphas),
            len(residuals),
            len(agg_modes),
            len(activations),
            len(biases),
            len(share_weights),
        ]
        assert len(set(lengths)) == 1, (
            "Expect the lengths of out_feats, num_heads, "
            "feat_drops, attn_drops, alphas, residuals, "
            "agg_modes, activations, and biases to be the same, "
            "got {}".format(lengths)
        )
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                GATv2Layer(
                    in_feats=in_feats,
                    out_feats=out_feats[i],
                    num_heads=num_heads[i],
                    feat_drop=feat_drops[i],
                    attn_drop=attn_drops[i],
                    negative_slope=alphas[i],
                    residual=residuals[i],
                    agg_mode=agg_modes[i],
                    activation=activations[i],
                    bias=biases[i],
                    allow_zero_in_degree=allow_zero_in_degree,
                    share_weights=share_weights[i],
                )
            )
            if agg_modes[i] == "flatten":
                in_feats = out_feats[i] * num_heads[i]
            else:
                in_feats = out_feats[i]

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
              hidden_sizes[-1] if agg_modes[-1] == 'mean' and
              hidden_sizes[-1] * num_heads[-1] otherwise.
        """
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats
