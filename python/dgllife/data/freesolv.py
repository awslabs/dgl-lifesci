# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# FreeSolv from MoleculeNet for the prediction of hydration free
# energy of small molecules in water

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

class FreeSolv(MoleculeCSVDataset):
    r"""FreeSolv from MoleculeNet for the prediction of hydration free
    energy of small molecules in water
    """
    def __init__(self):
        return NotImplementedError
