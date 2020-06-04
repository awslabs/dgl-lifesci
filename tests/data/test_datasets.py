# -*- coding: utf-8 -*-
#
# test_datasets.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from dgllife.data import *

def remove_file(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass

def test_alchemy():
    print('Test Alchemy')
    dataset = TencentAlchemyDataset(mode='valid',
                                    node_featurizer=None,
                                    edge_featurizer=None)
    dataset = TencentAlchemyDataset(mode='valid',
                                    node_featurizer=None,
                                    edge_featurizer=None,
                                    load=False)

def test_pdbbind():
    print('Test PDBBind')
    dataset = PDBBind(subset='core', remove_hs=True)

def test_pubchem_aromaticity():
    print('Test pubchem aromaticity')
    dataset = PubChemBioAssayAromaticity()
    remove_file('pubchem_aromaticity_dglgraph.bin')

def test_tox21():
    print('Test Tox21')
    dataset = Tox21()
    remove_file('tox21_dglgraph.bin')

if __name__ == '__main__':
    test_alchemy()
    test_pdbbind()
    test_pubchem_aromaticity()
    test_tox21()
