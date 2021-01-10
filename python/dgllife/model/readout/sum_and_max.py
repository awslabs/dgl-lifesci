# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Apply sum and max pooling to the node representations and concatenate the results.
# pylint: disable= no-member, arguments-differ, invalid-name

import dgl
import torch

__all__ = ['SumAndMax']

# pylint: disable=W0221, W0622
class SumAndMax(object):
    r"""Apply sum and max pooling to the node
    representations and concatenate the results.
    """
    def __init__(self):
        pass

    def forward(self, bg, feats):
        """Readout

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size.

        Returns
        -------
        h_g : FloatTensor of shape (B, 2 * M1)
            * B is the number of graphs in the batch
        """
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_sum = dgl.sum_nodes(bg, 'h')
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)
