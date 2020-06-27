# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from dgllife.model import HadamardLinkPredictor

def test_hadamard_link_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    num_pairs = 4
    in_feats = 2
    model = HadamardLinkPredictor(in_feats=in_feats, hidden_feats=3, num_layers=3).to(device)
    left_node_feats = torch.randn(num_pairs, in_feats).to(device)
    right_node_feats = torch.randn(num_pairs, in_feats).to(device)
    assert model(left_node_feats, right_node_feats).shape == torch.Size([num_pairs, 1])

if __name__ == '__main__':
    test_hadamard_link_predictor()
