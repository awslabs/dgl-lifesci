# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Utilities for using pretrained models.
# pylint: disable= no-member, arguments-differ, invalid-name

import torch

from dgl.data.utils import _get_dgl_url, download

from .moleculenet import *
from .generative_models import *
from .property_prediction import *
from .reaction import *

__all__ = ['load_pretrained']

url = {**moleculenet_url, **generative_url, **property_url, **reaction_url}

def download_and_load_checkpoint(model_name, model, model_postfix,
                                 local_pretrained_path='pre_trained.pth', log=True):
    """Download pretrained model checkpoint

    The model will be loaded to CPU.

    Parameters
    ----------
    model_name : str
        Name of the model
    model : nn.Module
        Instantiated model instance
    model_postfix : str
        Postfix for pretrained model checkpoint
    local_pretrained_path : str
        Local name for the downloaded model checkpoint
    log : bool
        Whether to print progress for model loading

    Returns
    -------
    model : nn.Module
        Pretrained model
    """
    url_to_pretrained = _get_dgl_url(model_postfix)
    local_pretrained_path = '_'.join([model_name, local_pretrained_path])
    download(url_to_pretrained, path=local_pretrained_path, log=log)
    checkpoint = torch.load(local_pretrained_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if log:
        print('Pretrained model loaded')

    return model

# pylint: disable=I1101
def load_pretrained(model_name, log=True):
    """Load a pretrained model

    Parameters
    ----------
    model_name : str
        Currently supported options include

        * ``'GCN_Tox21'``: A GCN-based model for molecular property prediction on Tox21
        * ``'GAT_Tox21'``: A GAT-based model for molecular property prediction on Tox21
        * ``'Weave_Tox21'``: A Weave model for molecular property prediction on Tox21
        * ``'AttentiveFP_Aromaticity'``: An AttentiveFP model for predicting number of
          aromatic atoms on a subset of Pubmed
        * ``'DGMG_ChEMBL_canonical'``: A DGMG model trained on ChEMBL with a canonical
          atom order
        * ``'DGMG_ChEMBL_random'``: A DGMG model trained on ChEMBL for molecule generation
          with a random atom order
        * ``'DGMG_ZINC_canonical'``: A DGMG model trained on ZINC for molecule generation
          with a canonical atom order
        * ``'DGMG_ZINC_random'``: A DGMG model pre-trained on ZINC for molecule generation
          with a random atom order
        * ``'JTNN_ZINC'``: A JTNN model pre-trained on ZINC for molecule generation
        * ``'wln_center_uspto'``: A WLN model pre-trained on USPTO for reaction prediction
        * ``'wln_rank_uspto'``: A WLN model pre-trained on USPTO for candidate product ranking
        * ``'gin_supervised_contextpred'``: A GIN model pre-trained with supervised learning
          and context prediction
        * ``'gin_supervised_infomax'``: A GIN model pre-trained with supervised learning
          and deep graph infomax
        * ``'gin_supervised_edgepred'``: A GIN model pre-trained with supervised learning
          and edge prediction
        * ``'gin_supervised_masking'``: A GIN model pre-trained with supervised learning
          and attribute masking
        * ``'GCN_canonical_MUV'``: A GCN model trained on MUV with canonical
          featurization for atoms
        * ``'GCN_attentivefp_MUV'``: A GCN model trained on MUV with attentivefp
          featurization for atoms
        * ``'GAT_canonical_MUV'``: A GAT model trained on MUV with canonical
          featurization for atoms
        * ``'GAT_attentivefp_MUV'``: A GAT model trained on MUV with attentivefp
          featurization for atoms
        * ``'Weave_canonical_MUV'``: A Weave model trained on MUV with canonical
          featurization for atoms
        * ``'Weave_attentivefp_MUV'``: A Weave model trained on MUV with attentivefp
          featurization for atoms
        * ``'MPNN_canonical_MUV'``: An MPNN model trained on MUV with canonical
          featurization for atoms
        * ``'MPNN_attentivefp_MUV'``: An MPNN model trained on MUV with attentivefp
          featurization for atoms

    log : bool
        Whether to print progress for model loading

    Returns
    -------
    model
    """
    if model_name not in url:
        raise RuntimeError("Cannot find a pretrained model with name {}".format(model_name))

    for func in [create_moleculenet_model, create_generative_model,
                 create_property_model, create_reaction_model]:
        model = func(model_name)
        if model is not None:
            break

    return download_and_load_checkpoint(model_name, model, url[model_name], log=log)
