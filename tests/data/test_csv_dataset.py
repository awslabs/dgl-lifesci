# -*- coding: utf-8 -*-
#
# test_csv_dataset.py
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
import pandas as pd

from dgllife.data.csv_dataset import *
from dgllife.utils.featurizers import *
from dgllife.utils.mol_to_graph import *

def test_data_frame():
    data = [['CCO', 0, 1], ['CO', 2, 3]]
    df = pd.DataFrame(data, columns = ['smiles', 'task1', 'task2'])

    return df

def remove_file(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass

def test_mol_csv():
    df = test_data_frame()
    fname = 'test.bin'
    dataset = MoleculeCSVDataset(df=df, smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=CanonicalAtomFeaturizer(),
                                 edge_featurizer=CanonicalBondFeaturizer(),
                                 smiles_column='smiles',
                                 cache_file_path=fname)
    assert dataset.task_names == ['task1', 'task2']
    smiles, graph, label, mask = dataset[0]
    assert label.shape[0] == 2
    assert mask.shape[0] == 2
    assert 'h' in graph.ndata
    assert 'e' in graph.edata

    # Test task_names
    dataset = MoleculeCSVDataset(df=df, smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=None,
                                 edge_featurizer=None,
                                 smiles_column='smiles',
                                 cache_file_path=fname,
                                 task_names=['task1'])
    assert dataset.task_names == ['task1']

    # Test load
    dataset = MoleculeCSVDataset(df=df, smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=CanonicalAtomFeaturizer(),
                                 edge_featurizer=None,
                                 smiles_column='smiles',
                                 cache_file_path=fname,
                                 load=True)
    smiles, graph, label, mask = dataset[0]
    assert 'h' in graph.ndata
    assert 'e' in graph.edata

    dataset = MoleculeCSVDataset(df=df, smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=CanonicalAtomFeaturizer(),
                                 edge_featurizer=None,
                                 smiles_column='smiles',
                                 cache_file_path=fname,
                                 load=False)
    smiles, graph, label, mask = dataset[0]
    assert 'h' in graph.ndata
    assert 'e' not in graph.edata

    remove_file(fname)

if __name__ == '__main__':
    test_mol_csv()
