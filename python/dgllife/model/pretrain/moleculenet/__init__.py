# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Index for pre-trained models on MoleculeNet

from .muv import *
from .tox21 import *

__all__ = ['moleculenet_url',
           'create_moleculenet_model']

moleculenet_url = {**muv_url, **tox21_url}

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
    for func in [create_muv_model, create_tox21_model]:
        model = func(model_name)
        if model is not None:
            return model
    return None
