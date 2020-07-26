# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dgllife.utils import analyze_mols

def remove_file(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass

def test_analyze_mols():
    smiles = ['CCO', 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C', '1']
    results = analyze_mols(smiles, path_to_export='.')
    assert results['num_atoms'] == [3, 23]
    assert results['num_bonds'] == [2, 25]
    assert results['num_rings'] == [0, 3]
    assert results['num_input_mols'] == 3
    assert results['num_valid_mols'] == 2
    assert results['valid_proportion'] == 0.6666666666666666
    assert results['cano_smi'] == [
        'CCO', 'CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O']
    assert results['atom_type_frequency'] == {'O': 2, 'C': 2, 'N': 1, 'S': 1}
    assert results['degree_frequency'] == {1: 2, 2: 2, 3: 1, 4: 1}
    assert results['total_degree_frequency'] == {2: 2, 4: 2, 1: 1, 3: 1}
    assert results['explicit_valence_frequency'] == {1: 2, 2: 2, 3: 1, 4: 1}
    assert results['implicit_valence_frequency'] == {1: 2, 2: 2, 3: 2, 0: 1}
    assert results['hybridization_frequency'] == {'SP3': 2, 'SP2': 1}
    assert results['total_num_h_frequency'] == {1: 2, 2: 2, 3: 2, 0: 1}
    assert results['formal_charge_frequency'] == {0: 2}
    assert results['num_radical_electrons_frequency'] == {0: 2}
    assert results['aromatic_atom_frequency'] == {False: 2, True: 1}
    assert results['chirality_tag_frequency'] == {'CHI_UNSPECIFIED': 2,
                                                  'CHI_TETRAHEDRAL_CCW': 1,
                                                  'CHI_TETRAHEDRAL_CW': 1}
    assert results['bond_type_frequency'] == {'SINGLE': 2, 'DOUBLE': 1, 'AROMATIC': 1}
    assert results['conjugated_bond_frequency'] == {False: 2, True: 1}
    assert results['bond_stereo_configuration_frequency'] == {'STEREONONE': 2}
    assert results['bond_direction_frequency'] == {'NONE': 2}

    remove_file('valid_canonical_smiles.txt')
    remove_file('summary.txt')

if __name__ == '__main__':
    test_analyze_mols()
