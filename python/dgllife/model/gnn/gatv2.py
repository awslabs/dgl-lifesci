# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Attention Networks v2
#
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv

__all__ = ["GATv2"]

# pylint: disable=W0221
class GATv2Layer(nn.Module):
    r"""Single GATv2 layer from `How Attentive Are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`

    Parameters
    ----------
    in_feats : int
        Number of input node features
    out_feats : int
        Number of output node features
    num_heads : int
        Number of attention heads
    feat_drop : float, optional
        Dropout rate on feature. Defaults: 0
    attn_drop : float, optional
        Dropout rate on attention values of edges. Defaults: 0
    negative_slope : float, optional
        Hyperparameter in LeakyReLU, which is the slope for negative values.
        Default to 0.2.
    residual : bool, optional
        If True, use residual connection. Defaults: False.
    activation : callable, optional
        If not None, the activation function will be applied to the updated
        node features. Default: None.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes
        will be invalid since no messages will be passed to those nodes. This
        is harmful for some applications, causing silent performance regression
        . This module will raise a DGLError if it detects 0-in-degree nodes in
        input graph. By setting True, it will suppress the check and let the
        users handle it by themselves. Defaults: False.
    bias : bool, optional
        If set to False, the layer will not learn an additive bias.
        Defaults: True.
    share_weights : bool, optional
        If set to True, the learnable weight matrix for source and destination
        nodes will be the same. Defaults: False.
    agg_mode : str
        The way to aggregate multi-head attention results, can be either
        'flatten' for concatenating all-head results or 'mean' for averaging
        all head results.
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

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gatv2_conv.reset_parameters()

    def forward(self, bg, feats, get_attention=False):
        """Update node representations

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs.
            * M1 is the input node feature size, which equals in_feats in
              initialization
        get_attention : bool, optional
            Whether to return the attention values. Defaults: False

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs.
            * M2 is the output node representation size, which equals
              out_feats in initialization if self.agg_mode == 'mean' and
              out_feats * num_heads otherwise.
        attention : FloatTensor of shape (E, H, 1), optional
            Attention values, returned when :attr:`get_attention` is True

            * `E` is the number of edges.
            * `H` is the number of attention heads.
        """
        if get_attention:
            out_feats, attention = self.gatv2_conv(
                bg, feats, get_attention=True
            )
        else:
            out_feats = self.gatv2_conv(bg, feats)

        if self.agg_mode == "flatten":
            out_feats = out_feats.flatten(1)
        else:
            out_feats = out_feats.mean(1)

        if get_attention:
            return out_feats, attention
        else:
            return out_feats


class GATv2(nn.Module):
    r"""GATv2 from `How Attentive Are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`

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
    ):
        super(GATv2, self).__init__()

        if hidden_feats is None:
            hidden_feats = [32, 32]

        n_layers = len(hidden_feats)
        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0.0 for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0.0 for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [False for _ in range(n_layers)]
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        if biases is None:
            biases = [True for _ in range(n_layers)]
        if share_weights is None:
            share_weights = [False for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ["flatten" for _ in range(n_layers - 1)]
            agg_modes.append("mean")

        lengths = [
            len(hidden_feats),
            len(num_heads),
            len(feat_drops),
            len(attn_drops),
            len(alphas),
            len(residuals),
            len(activations),
            len(biases),
            len(share_weights),
            len(agg_modes),
        ]
        assert len(set(lengths)) == 1, (
            "Expect the lengths of hidden_feats, num_heads, feat_drops, "
            "attn_drops, alphas, residuals, activations, biases, "
            "share_weights, and agg_modes to be the same, "
            "got {}".format(lengths)
        )
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                GATv2Layer(
                    in_feats=in_feats,
                    out_feats=hidden_feats[i],
                    num_heads=num_heads[i],
                    feat_drop=feat_drops[i],
                    attn_drop=attn_drops[i],
                    negative_slope=alphas[i],
                    residual=residuals[i],
                    activation=activations[i],
                    allow_zero_in_degree=allow_zero_in_degree,
                    bias=biases[i],
                    share_weights=share_weights[i],
                    agg_mode=agg_modes[i],
                )
            )
            if agg_modes[i] == "flatten":
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, bg, feats, get_attention=False):
        """Update node representations.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs.
            * M1 is the input node feature size, which equals in_feats in
              initialization
        get_attention : bool, optional
            Whether to return the attention values. Defaults: False

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs.
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] if agg_modes[-1] == 'mean' and
              hidden_sizes[-1] * num_heads[-1] otherwise.
        attentions : list of FloatTensor of shape (E, H, 1), optional
            It is returned when :attr:`get_attention` is True.
            ``attentions[i]`` gives the attention values in the i-th GATv2
            layer.

            * `E` is the number of edges.
            * `H` is the number of attention heads.
        """
        if get_attention:
            attentions = []
            for gnn in self.gnn_layers:
                feats, attention = gnn(bg, feats, get_attention=get_attention)
                attentions.append(attention)
            return feats, attentions
        else:
            for gnn in self.gnn_layers:
                feats = gnn(bg, feats, get_attention=get_attention)
            return feats
