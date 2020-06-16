# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import shutil

from dgl.data.utils import download, _get_dgl_url, extract_archive
from dgllife.utils.io import get_mol_3d_coordinates, load_molecule, load_smiles_from_txt
from rdkit import Chem
from rdkit.Chem import AllChem

def test_get_mol_3D_coordinates():
    mol = Chem.MolFromSmiles('CCO')
    # Test the case when conformation does not exist
    assert get_mol_3d_coordinates(mol) is None

    # Test the case when conformation exists
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    coords = get_mol_3d_coordinates(mol)
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (mol.GetNumAtoms(), 3)

def remove_dir(dir):
    if os.path.isdir(dir):
        try:
            shutil.rmtree(dir)
        except OSError:
            pass

def test_load_molecule():
    remove_dir('tmp1')
    remove_dir('tmp2')

    url = _get_dgl_url('dgllife/example_mols.tar.gz')
    local_path = 'tmp1/example_mols.tar.gz'
    download(url, path=local_path)
    extract_archive(local_path, 'tmp2')

    load_molecule('tmp2/example_mols/example.sdf')
    load_molecule('tmp2/example_mols/example.mol2', use_conformation=False, sanitize=True)
    load_molecule('tmp2/example_mols/example.pdbqt', calc_charges=True)
    mol, _ = load_molecule('tmp2/example_mols/example.pdb', remove_hs=True)
    assert mol.GetNumAtoms() == mol.GetNumHeavyAtoms()

    remove_dir('tmp1')
    remove_dir('tmp2')

def test_load_smiles_from_txt():
    smiles_list1 = ['CCO', 'O=P(O)(OC1O[C@@H]([C@@H](O)[C@H](O)[C@H]1O)CO)O']
    file = 'smiles.txt'
    with open(file, 'w') as f:
        for s in smiles_list1:
            f.write(s + '\n')
    smiles_list2 = load_smiles_from_txt(file)
    assert smiles_list1 == smiles_list2

if __name__ == '__main__':
    test_get_mol_3D_coordinates()
    test_load_molecule()
    test_load_smiles_from_txt()
