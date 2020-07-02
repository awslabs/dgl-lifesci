# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lipophilicity from MoleculeNet for the prediction of octanol/water
# distribution coefficient (logD at pH 7.4) of 4200 compounds

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

class Lipophilicity(MoleculeCSVDataset):
    r"""Lipophilicity from MoleculeNet for the prediction of octanol/water
    distribution coefficient (logD at pH 7.4) of 4200 compounds
    """
    def __init__(self):
        return NotImplementedError
