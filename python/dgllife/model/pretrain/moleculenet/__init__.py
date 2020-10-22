# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Index for pre-trained models on MoleculeNet

from .bace import *
from .bbbp import *
from .clintox import *
from .esol import *
from .freesolv import *
from .hiv import *
from .lipophilicity import *
from .pcba import *
from .muv import *
from .sider import *
from .tox21 import *
from .toxcast import *

__all__ = ['moleculenet_url',
           'create_moleculenet_model']

moleculenet_url = {**bace_url, **bbbp_url, **clintox_url, **esol_url, **freesolv_url, **hiv_url,
                   **lipophilicity_url, **muv_url, **pcba_url, **sider_url, **tox21_url,
                   **toxcast_url}

def create_moleculenet_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    for func in [create_bace_model, create_bbbp_model, create_clintox_model, create_esol_model,
                 create_freesolv_model, create_hiv_model, create_lipophilicity_model,
                 create_muv_model, create_pcba_model, create_sider_model, create_tox21_model,
                 create_toxcast_model]:
        model = func(model_name)
        if model is not None:
            return model
    return None
