# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Utilities for using pre-trained models.

import torch

from dgl.data.utils import _get_dgl_url, download

from .moleculenet import *
from .generative_models import *
from .property_prediction import *
from .reaction import *

__all__ = ['load_pretrained']

url = {**moleculenet_url, **generative_url, **property_url, **reaction_url}

# pylint: disable=W0702
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
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model.load_state_dict(checkpoint)

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
        * ``'JTVAE_ZINC_no_kl'``: A JTVAE pre-trained on ZINC for molecule generation,
          without KL regularization
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
        * ``'GCN_canonical_BACE'``: A GCN model trained on BACE with canonical
          featurization for atoms
        * ``'GCN_attentivefp_BACE'``: A GCN model trained on BACE with attentivefp
          featurization for atoms
        * ``'GAT_canonical_BACE'``: A GAT model trained on BACE with canonical
          featurization for atoms
        * ``'GAT_attentivefp_BACE'``: A GAT model trained on BACE with attentivefp
          featurization for atoms
        * ``'Weave_canonical_BACE'``: A Weave model trained on BACE with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_BACE'``: A Weave model trained on BACE with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_BACE'``: An MPNN model trained on BACE with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_BACE'``: An MPNN model trained on BACE with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_BACE'``: An AttentiveFP model trained on BACE with
          canonical featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_BACE'``: An AttentiveFP model trained on BACE with
          attentivefp featurization for atoms and bonds
        * ``'gin_supervised_contextpred_BACE'``: A GIN model pre-trained with supervised
          learning and context prediction, and fine-tuned on BACE
        * ``'gin_supervised_infomax_BACE'``: A GIN model pre-trained with supervised learning
          and infomax, and fine-tuned on BACE
        * ``'gin_supervised_edgepred_BACE'``: A GIN model pre-trained with supervised learning
          and edge prediction, and fine-tuned on BACE
        * ``'gin_supervised_masking_BACE'``: A GIN model pre-trained with supervised learning
          and masking, and fine-tuned on BACE
        * ``'NF_canonical_BACE'``: An NF model trained on BACE with canonical
          featurization for atoms
        * ``'GCN_canonical_BBBP'``: A GCN model trained on BBBP with canonical
          featurization for atoms
        * ``'GCN_attentivefp_BBBP'``: A GCN model trained on BBBP with attentivefp
          featurization for atoms
        * ``'GAT_canonical_BBBP'``: A GAT model trained on BBBP with canonical
          featurization for atoms
        * ``'GAT_attentivefp_BBBP'``: A GAT model trained on BBBP with attentivefp
          featurization for atoms
        * ``'Weave_canonical_BBBP'``: A Weave model trained on BBBP with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_BBBP'``: A Weave model trained on BBBP with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_BBBP'``: An MPNN model trained on BBBP with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_BBBP'``: An MPNN model trained on BBBP with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_BBBP'``: An AttentiveFP model trained on BBBP with
          canonical featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_BBBP'``: An AttentiveFP model trained on BBBP with
          attentivefp featurization for atoms and bonds
        * ``'gin_supervised_contextpred_BBBP'``: A GIN model pre-trained with supervised
          learning and context prediction, and fine-tuned on BBBP
        * ``'gin_supervised_infomax_BBBP'``: A GIN model pre-trained with supervised learning
          and infomax, and fine-tuned on BBBP
        * ``'gin_supervised_edgepred_BBBP'``: A GIN model pre-trained with supervised learning
          and edge prediction, and fine-tuned on BBBP
        * ``'gin_supervised_masking_BBBP'``: A GIN model pre-trained with supervised learning
          and masking, and fine-tuned on BBBP
        * ``'NF_canonical_BBBP'``: An NF model pre-trained on BBBP with canonical
          featurization for atoms
        * ``'GCN_canonical_ClinTox'``: A GCN model trained on ClinTox with canonical
          featurization for atoms
        * ``'GCN_attentivefp_ClinTox'``: A GCN model trained on ClinTox with attentivefp
          featurization for atoms
        * ``'GAT_canonical_ClinTox'``: A GAT model trained on ClinTox with canonical
          featurization for atoms
        * ``'GAT_attentivefp_ClinTox'``: A GAT model trained on ClinTox with attentivefp
          featurization for atoms
        * ``'Weave_canonical_ClinTox'``: A Weave model trained on ClinTox with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_ClinTox'``: A Weave model trained on ClinTox with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_ClinTox'``: An MPNN model trained on ClinTox with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_ClinTox'``: An MPNN model trained on ClinTox with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_ClinTox'``: An AttentiveFP model trained on ClinTox with
          canonical featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_BACE'``: An AttentiveFP model trained on ClinTox with
          attentivefp featurization for atoms and bonds
        * ``'GCN_canonical_ESOL'``: A GCN model trained on ESOL with canonical
          featurization for atoms
        * ``'GCN_attentivefp_ESOL'``: A GCN model trained on ESOL with attentivefp
          featurization for atoms
        * ``'GAT_canonical_ESOL'``: A GAT model trained on ESOL with canonical
          featurization for atoms
        * ``'GAT_attentivefp_ESOL'``: A GAT model trained on ESOL with attentivefp
          featurization for atoms
        * ``'Weave_canonical_ESOL'``: A Weave model trained on ESOL with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_ESOL'``: A Weave model trained on ESOL with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_ESOL'``: An MPNN model trained on ESOL with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_ESOL'``: An MPNN model trained on ESOL with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_ESOL'``: An AttentiveFP model trained on ESOL with
          canonical featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_ESOL'``: An AttentiveFP model trained on ESOL with
          attentivefp featurization for atoms and bonds
        * ``'gin_supervised_contextpred_ESOL'``: A GIN model pre-trained with supervised
          learning and context prediction, and fine-tuned on ESOL
        * ``'gin_supervised_infomax_ESOL'``: A GIN model pre-trained with supervised learning
          and infomax, and fine-tuned on ESOL
        * ``'gin_supervised_edgepred_ESOL'``: A GIN model pre-trained with supervised learning
          and edge prediction, and fine-tuned on ESOL
        * ``'gin_supervised_masking_ESOL'``: A GIN model pre-trained with supervised learning
          and masking, and fine-tuned on ESOL
        * ``'GCN_canonical_FreeSolv'``: A GCN model trained on FreeSolv with canonical
          featurization for atoms
        * ``'GCN_attentivefp_FreeSolv'``: A GCN model trained on FreeSolv with attentivefp
          featurization for atoms
        * ``'GAT_canonical_FreeSolv'``: A GAT model trained on FreeSolv with canonical
          featurization for atoms
        * ``'GAT_attentivefp_FreeSolv'``: A GAT model trained on FreeSolv with attentivefp
          featurization for atoms
        * ``'Weave_canonical_FreeSolv'``: A Weave model trained on FreeSolv with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_FreeSolv'``: A Weave model trained on FreeSolv with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_FreeSolv'``: An MPNN model trained on FreeSolv with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_FreeSolv'``: An MPNN model trained on FreeSolv with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_FreeSolv'``: An AttentiveFP model trained on FreeSolv with
          canonical featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_FreeSolv'``: An AttentiveFP model trained on FreeSolv with
          attentivefp featurization for atoms and bonds
        * ``'gin_supervised_contextpred_FreeSolv'``: A GIN model pre-trained with supervised
          learning and context prediction, and fine-tuned on FreeSolv
        * ``'gin_supervised_infomax_FreeSolv'``: A GIN model pre-trained with supervised learning
          and infomax, and fine-tuned on FreeSolv
        * ``'gin_supervised_edgepred_FreeSolv'``: A GIN model pre-trained with supervised learning
          and edge prediction, and fine-tuned on FreeSolv
        * ``'gin_supervised_masking_FreeSolv'``: A GIN model pre-trained with supervised learning
          and masking, and fine-tuned on FreeSolv
        * ``'GCN_canonical_HIV'``: A GCN model trained on HIV with canonical
          featurization for atoms
        * ``'GCN_attentivefp_HIV'``: A GCN model trained on HIV with attentivefp
          featurization for atoms
        * ``'GAT_canonical_HIV'``: A GAT model trained on BACE with canonical
          featurization for atoms
        * ``'GAT_attentivefp_HIV'``: A GAT model trained on BACE with attentivefp
          featurization for atoms
        * ``'Weave_canonical_HIV'``: A Weave model trained on HIV with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_HIV'``: A Weave model trained on HIV with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_HIV'``: An MPNN model trained on HIV with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_HIV'``: An MPNN model trained on HIV with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_HIV'``: An AttentiveFP model trained on HIV with canonical
          featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_HIV'``: An AttentiveFP model trained on HIV with attentivefp
          featurization for atoms and bonds
        * ``'gin_supervised_contextpred_HIV'``: A GIN model pre-trained with supervised learning
          and context prediction, and fine-tuned on HIV
        * ``'gin_supervised_infomax_HIV'``: A GIN model pre-trained with supervised learning
          and infomax, and fine-tuned on HIV
        * ``'gin_supervised_edgepred_HIV'``: A GIN model pre-trained with supervised learning
          and edge prediction, and fine-tuned on HIV
        * ``'gin_supervised_masking_HIV'``: A GIN model pre-trained with supervised learning
          and masking, and fine-tuned on HIV
        * ``'NF_canonical_HIV'``: An NF model trained on HIV with canonical
          featurization for atoms
        * ``'GCN_canonical_Lipophilicity'``: A GCN model trained on Lipophilicity with canonical
          featurization for atoms
        * ``'GCN_attentivefp_Lipophilicity'``: A GCN model trained on Lipophilicity with
          attentivefp featurization for atoms
        * ``'GAT_canonical_Lipophilicity'``: A GAT model trained on Lipophilicity with canonical
          featurization for atoms
        * ``'GAT_attentivefp_Lipophilicity'``: A GAT model trained on Lipophilicity with
          attentivefp featurization for atoms
        * ``'Weave_canonical_Lipophilicity'``: A Weave model trained on Lipophilicity with
          canonical featurization for atoms and bonds
        * ``'Weave_attentivefp_Lipophilicity'``: A Weave model trained on Lipophilicity with
          attentivefp featurization for atoms and bonds
        * ``'MPNN_canonical_Lipophilicity'``: An MPNN model trained on Lipophilicity with
          canonical featurization for atoms and bonds
        * ``'MPNN_attentivefp_Lipophilicity'``: An MPNN model trained on Lipophilicity with
          attentivefp featurization for atoms and bonds
        * ``'AttentiveFP_canonical_Lipophilicity'``: An AttentiveFP model trained on
          Lipophilicity with canonical featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_Lipophilicity'``: An AttentiveFP model trained on
          Lipophilicity with attentivefp featurization for atoms and bonds
        * ``'gin_supervised_contextpred_Lipophilicity'``: A GIN model pre-trained with supervised
          learning and context prediction, and fine-tuned on Lipophilicity
        * ``'gin_supervised_infomax_Lipophilicity'``: A GIN model pre-trained with supervised
          learning and infomax, and fine-tuned on Lipophilicity
        * ``'gin_supervised_edgepred_Lipophilicity'``: A GIN model pre-trained with supervised
          learning and edge prediction, and fine-tuned on Lipophilicity
        * ``'gin_supervised_masking_Lipophilicity'``: A GIN model pre-trained with supervised
          learning and masking, and fine-tuned on Lipophilicity
        * ``'GCN_canonical_MUV'``: A GCN model trained on MUV with canonical
          featurization for atoms
        * ``'GCN_attentivefp_MUV'``: A GCN model trained on MUV with attentivefp
          featurization for atoms
        * ``'GAT_canonical_MUV'``: A GAT model trained on MUV with canonical
          featurization for atoms
        * ``'GAT_attentivefp_MUV'``: A GAT model trained on MUV with attentivefp
          featurization for atoms
        * ``'Weave_canonical_MUV'``: A Weave model trained on MUV with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_MUV'``: A Weave model trained on MUV with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_MUV'``: An MPNN model trained on MUV with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_MUV'``: An MPNN model trained on MUV with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_MUV'``: An AttentiveFP model trained on MUV with canonical
          featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_MUV'``: An AttentiveFP model trained on MUV with attentivefp
          featurization for atoms and bonds
        * ``'gin_supervised_contextpred_MUV'``: A GIN model pre-trained with supervised learning
          and context prediction, and fine-tuned on MUV
        * ``'gin_supervised_infomax_MUV'``: A GIN model pre-trained with supervised learning
          and infomax, and fine-tuned on MUV
        * ``'gin_supervised_edgepred_MUV'``: A GIN model pre-trained with supervised learning
          and edge prediction, and fine-tuned on MUV
        * ``'gin_supervised_masking_MUV'``: A GIN model pre-trained with supervised learning
          and masking, and fine-tuned on MUV
        * ``'GCN_canonical_PCBA'``: A GCN model trained on PCBA with canonical
          featurization for atoms
        * ``'GCN_attentivefp_PCBA'``: A GCN model trained on PCBA with attentivefp
          featurization for atoms
        * ``'GAT_canonical_PCBA'``: A GAT model trained on PCBA with canonical
          featurization for atoms
        * ``'GAT_attentivefp_PCBA'``: A GAT model trained on PCBA with attentivefp
          featurization for atoms
        * ``'Weave_canonical_PCBA'``: A Weave model trained on PCBA with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_PCBA'``: A Weave model trained on PCBA with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_PCBA'``: An MPNN model trained on PCBA with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_PCBA'``: An MPNN model trained on PCBA with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_PCBA'``: An AttentiveFP model trained on PCBA with
          canonical featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_PCBA'``: An AttentiveFP model trained on PCBA with
          attentivefp featurization for atoms and bonds
        * ``'GCN_canonical_SIDER'``: A GCN model trained on SIDER with canonical
          featurization for atoms
        * ``'GCN_attentivefp_SIDER'``: A GCN model trained on SIDER with attentivefp
          featurization for atoms
        * ``'GAT_canonical_SIDER'``: A GAT model trained on SIDER with canonical
          featurization for atoms
        * ``'GAT_attentivefp_SIDER'``: A GAT model trained on SIDER with attentivefp
          featurization for atoms
        * ``'Weave_canonical_SIDER'``: A Weave model trained on SIDER with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_SIDER'``: A Weave model trained on SIDER with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_SIDER'``: An MPNN model trained on SIDER with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_SIDER'``: An MPNN model trained on SIDER with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_SIDER'``: An AttentiveFP model trained on SIDER with
          canonical featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_SIDER'``: An AttentiveFP model trained on SIDER with
          attentivefp featurization for atoms and bonds
        * ``'gin_supervised_contextpred_SIDER'``: A GIN model pre-trained with supervised learning
          and context prediction, and fine-tuned on SIDER
        * ``'gin_supervised_infomax_SIDER'``: A GIN model pre-trained with supervised learning
          and infomax, and fine-tuned on SIDER
        * ``'gin_supervised_edgepred_SIDER'``: A GIN model pre-trained with supervised learning
          and edge prediction, and fine-tuned on SIDER
        * ``'gin_supervised_masking_SIDER'``: A GIN model pre-trained with supervised learning
          and masking, and fine-tuned on SIDER
        * ``'NF_canonical_SIDER'``: An NF model trained on SIDER with canonical
          featurization for atoms
        * ``'GCN_canonical_Tox21'``: A GCN model trained on Tox21 with canonical
          featurization for atoms
        * ``'GCN_attentivefp_Tox21'``: A GCN model trained on Tox21 with attentivefp
          featurization for atoms
        * ``'GAT_canonical_Tox21'``: A GAT model trained on Tox21 with canonical
          featurization for atoms
        * ``'GAT_attentivefp_Tox21'``: A GAT model trained on Tox21 with attentivefp
          featurization for atoms
        * ``'Weave_canonical_Tox21'``: A Weave model trained on Tox21 with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_Tox21'``: A Weave model trained on Tox21 with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_Tox21'``: An MPNN model trained on Tox21 with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_Tox21'``: An MPNN model trained on Tox21 with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_Tox21'``: An AttentiveFP model trained on Tox21 with
          canonical featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_Tox21'``: An AttentiveFP model trained on Tox21 with
          attentivefp featurization for atoms and bonds
        * ``'gin_supervised_contextpred_Tox21'``: A GIN model pre-trained with supervised
          learning and context prediction, and fine-tuned on Tox21
        * ``'gin_supervised_infomax_Tox21'``: A GIN model pre-trained with supervised learning
          and infomax, and fine-tuned on Tox21
        * ``'gin_supervised_edgepred_Tox21'``: A GIN model pre-trained with supervised learning
          and edge prediction, and fine-tuned on Tox21
        * ``'gin_supervised_masking_Tox21'``: A GIN model pre-trained with supervised learning
          and masking, and fine-tuned on Tox21
        * ``'NF_canonical_Tox21'``: An NF model trained on Tox21 with canonical
          featurization for atoms
        * ``'GCN_canonical_ToxCast'``: A GCN model trained on ToxCast with canonical
          featurization for atoms
        * ``'GCN_attentivefp_ToxCast'``: A GCN model trained on ToxCast with attentivefp
          featurization for atoms
        * ``'GAT_canonical_ToxCast'``: A GAT model trained on ToxCast with canonical
          featurization for atoms
        * ``'GAT_attentivefp_ToxCast'``: A GAT model trained on ToxCast with attentivefp
          featurization for atoms
        * ``'Weave_canonical_ToxCast'``: A Weave model trained on ToxCast with canonical
          featurization for atoms and bonds
        * ``'Weave_attentivefp_ToxCast'``: A Weave model trained on ToxCast with attentivefp
          featurization for atoms and bonds
        * ``'MPNN_canonical_ToxCast'``: An MPNN model trained on ToxCast with canonical
          featurization for atoms and bonds
        * ``'MPNN_attentivefp_ToxCast'``: An MPNN model trained on ToxCast with attentivefp
          featurization for atoms and bonds
        * ``'AttentiveFP_canonical_ToxCast'``: An AttentiveFP model trained on ToxCast with
          canonical featurization for atoms and bonds
        * ``'AttentiveFP_attentivefp_ToxCast'``: An AttentiveFP model trained on ToxCast with
          attentivefp featurization for atoms and bonds
        * ``'gin_supervised_contextpred_ToxCast'``: A GIN model pre-trained with supervised
          learning and context prediction, and fine-tuned on ToxCast
        * ``'gin_supervised_infomax_ToxCast'``: A GIN model pre-trained with supervised learning
          and infomax, and fine-tuned on ToxCast
        * ``'gin_supervised_edgepred_ToxCast'``: A GIN model pre-trained with supervised learning
          and edge prediction, and fine-tuned on ToxCast
        * ``'gin_supervised_masking_ToxCast'``: A GIN model pre-trained with supervised learning
          and masking, and fine-tuned on ToxCast
        * ``'NF_canonical_ToxCast'``: An NF model trained on ToxCast with canonical
          featurization for atoms and bonds

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
