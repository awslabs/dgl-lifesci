# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Utils for analyzing a collection of molecules

from collections import defaultdict
from multiprocessing import Pool

import itertools
import numpy as np

from rdkit import Chem

__all__ = ['analyze_mols']

def summarize_a_mol(mol):
    """Summarize a molecule

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Input molecule

    Returns
    -------
    dict
        Summarized info for a molecule
    """
    summary = {
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': len(Chem.GetSymmSSSR(mol)),
        'atom_type': set(),
        'degree': set(),
        'total_degree': set(),
        'explicit_valence': set(),
        'implicit_valence': set(),
        'hybridization': set(),
        'total_num_h': set(),
        'formal_charge': set(),
        'num_radical_electrons': set(),
        'aromatic_atom': set(),
        'chirality_tag': set(),
        'bond_type': set(),
        'conjugated_bond': set(),
        'bond_stereo_configuration': set(),
        'bond_direction': set()
    }

    for atom in mol.GetAtoms():
        summary['atom_type'].add(atom.GetSymbol())
        summary['degree'].add(atom.GetDegree())
        summary['total_degree'].add(atom.GetTotalDegree())
        summary['explicit_valence'].add(atom.GetExplicitValence())
        summary['implicit_valence'].add(atom.GetImplicitValence())
        summary['hybridization'].add(str(atom.GetHybridization()))
        summary['total_num_h'].add(atom.GetTotalNumHs())
        summary['formal_charge'].add(atom.GetFormalCharge())
        summary['num_radical_electrons'].add(atom.GetNumRadicalElectrons())
        summary['aromatic_atom'].add(atom.GetIsAromatic())
        summary['chirality_tag'].add(str(atom.GetChiralTag()))

    for bond in mol.GetBonds():
        summary['bond_type'].add(str(bond.GetBondType()))
        summary['conjugated_bond'].add(bond.GetIsConjugated())
        summary['bond_stereo_configuration'].add(str(bond.GetStereo()))
        summary['bond_direction'].add(str(bond.GetBondDir()))

    return summary

def count_frequency(values):
    """Count how many times each value appear

    Parameters
    ----------
    values : list

    Returns
    -------
    dict
        Mapping each unique value to the times it appears
    """
    frequency = defaultdict(int)
    for val in values:
        frequency[val] += 1
    return dict(frequency)

def analyze_mols(smiles=None, mols=None, num_processes=1, path_to_export=None):
    r"""Analyze a collection of molecules

    The analysis will 1) filter out invalid molecules and record the valid ones;
    2) record the number of molecules having each particular descriptor/element
    (e.g. single bond). The descriptors/elements considered include:


    If ``path_to_export`` is not None, we will export the analysis results to
    the following files in ``path_to_export``:

        * **valid_canonical_smiles.txt**: A file of canonical SMILES for valid molecules
        * **summary.txt**: A file of all analysis results,
          see the **Examples** section for more details. For summary, we either compute
          mean/std of values or count the frequency that a value appears in molecules.

    Parameters
    ----------
    smiles : list of str, optional
        SMILES strings for a collection of molecules. Can be omitted if mols is not None.
        (Default: None)
    mols : list of rdkit.Chem.rdchem.Mol objects, optional
        RDKit molecule instances for a collection of molecules.
        Can be omitted if smiles is not None. (Default: None)
    num_processes : int, optional
        Number of processes for data analysis. (Default: 1)
    path_to_export : str, optional
        The directory to export analysis results. If not None, we will export the analysis
        results to local files in the specified directory. (Default: None)

    Returns
    -------
    dict
        Summary of the analysis results. For more details, see the **Examples** section.

    Examples
    --------

    >>> from dgllife.utils import analyze_mols

    >>> smiles = ['CCO', 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C', '1']
    >>> # Analyze the results and save the results to the current directory
    >>> results = analyze_mols(smiles, path_to_export='.')
    >>> results
    {'num_atoms': [3, 23],                    # Number of atoms in each molecule
     'num_bonds': [2, 25],                    # Number of bonds in each molecule
     'num_rings': [0, 3],                     # Number of rings in each molecule
     'num_input_mols': 3,                     # Number of input molecules
     'num_valid_mols': 2,                     # Number of valid molecules
     'valid_proportion': 0.6666666666666666,  # Proportion of valid molecules
     'cano_smi': ['CCO',                      # Canonical SMILES for valid molecules
     'CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O'],

     # The following items give the number of times each descriptor value appears in molecules

     'atom_type_frequency': {'O': 2, 'C': 2, 'N': 1, 'S': 1},
     'degree_frequency': {1: 2, 2: 2, 3: 1, 4: 1},
     'total_degree_frequency': {2: 2, 4: 2, 1: 1, 3: 1},
     'explicit_valence_frequency': {1: 2, 2: 2, 3: 1, 4: 1},
     'implicit_valence_frequency': {1: 2, 2: 2, 3: 2, 0: 1},
     'hybridization_frequency': {'SP3': 2, 'SP2': 1},
     'total_num_h_frequency': {1: 2, 2: 2, 3: 2, 0: 1},
     'formal_charge_frequency': {0: 2},
     'num_radical_electrons_frequency': {0: 2},
     'aromatic_atom_frequency': {False: 2, True: 1},
     'chirality_tag_frequency': {'CHI_UNSPECIFIED': 2,
     'CHI_TETRAHEDRAL_CCW': 1,
     'CHI_TETRAHEDRAL_CW': 1},
     'bond_type_frequency': {'SINGLE': 2, 'DOUBLE': 1, 'AROMATIC': 1},
     'conjugated_bond_frequency': {False: 2, True: 1},
     'bond_stereo_configuration_frequency': {'STEREONONE': 2},
     'bond_direction_frequency': {'NONE': 2}}
    """
    assert not ((smiles is None) and (mols is None)), \
        'At least one of the smiles and mols should not be None'
    assert num_processes >= 1, \
        'Expect num_processes to be no smaller than 1, got {:d}'.format(num_processes)

    general_items = ['num_atoms', 'num_bonds', 'num_rings']
    atom_items = ['atom_type', 'degree', 'total_degree', 'explicit_valence',
                  'implicit_valence', 'hybridization', 'total_num_h', 'formal_charge',
                  'num_radical_electrons', 'aromatic_atom', 'chirality_tag']
    bond_items = ['bond_type', 'conjugated_bond', 'bond_stereo_configuration', 'bond_direction']

    # Holder of the analysis results
    summary = dict()
    for item in list(itertools.chain.from_iterable([general_items, atom_items, bond_items])):
        summary[item] = []

    # Check validity
    if mols is None:
        mols = []
        for smi in smiles:
            smi_mol = Chem.MolFromSmiles(smi)
            if smi_mol is not None:
                mols.append(smi_mol)
        summary['num_input_mols'] = len(smiles)
        summary['num_valid_mols'] = len(mols)
    else:
        summary['num_input_mols'] = len(mols)
        mols_ = []
        for smi_mol in mols:
            if smi_mol is not None:
                mols_.append(smi_mol)
        mols = mols_
        summary['num_valid_mols'] = len(mols)
    summary['valid_proportion'] = summary['num_valid_mols'] / summary['num_input_mols']

    # Get canonicalized SMILES
    summary['cano_smi'] = [Chem.MolToSmiles(smi_mol) for smi_mol in mols]

    if num_processes == 1:
        summary_per_mol = []
        for smi_mol in mols:
            summary_per_mol.append(summarize_a_mol(smi_mol))
    else:
        with Pool(processes=num_processes) as pool:
            summary_per_mol = pool.map(summarize_a_mol, mols)

    for mol_summary in summary_per_mol:
        for key in general_items:
            summary[key].append(mol_summary[key])
        for key in list(itertools.chain.from_iterable([atom_items, bond_items])):
            summary[key].extend(list(mol_summary[key]))

    for item in list(itertools.chain.from_iterable([atom_items, bond_items])):
        summary[item + '_frequency'] = count_frequency(summary[item])
        del summary[item]

    if path_to_export is None:
        return summary

    # Export the analysis results to local files
    with open(path_to_export + '/valid_canonical_smiles.txt', 'w') as file:
        for smi in summary['cano_smi']:
            file.write(smi + '\n')

    with open(path_to_export + '/summary.txt', 'w') as file:
        file.write('General statistics\n')
        file.write('=' * 60 + '\n')
        file.write('Number of input molecules: {}\n'.format(summary['num_input_mols']))
        file.write('Number of valid molecules: {}\n'.format(summary['num_valid_mols']))
        file.write('Percentage of valid molecules: {} %\n'.format(
            summary['valid_proportion'] * 100))
        file.write('Average number of atoms per molecule: {} +- {}\n'.format(
            np.mean(summary['num_atoms']), np.std(summary['num_atoms'])))
        file.write('Average number of bonds per molecule: {} +- {}\n'.format(
            np.mean(summary['num_bonds']), np.std(summary['num_bonds'])))
        file.write('Average number of rings per molecule: {} +- {}\n'.format(
            np.mean(summary['num_rings']), np.std(summary['num_rings'])))
        file.write('\n')

        file.write('Atom statistics\n')
        file.write('=' * 60 + '\n')
        for item in atom_items:
            file.write('{} frequency: {}\n'.format(item, summary[item + '_frequency']))
        file.write('\n')

        file.write('Bond statistics\n')
        file.write('=' * 60 + '\n')
        for item in bond_items:
            file.write('{} frequency: {}\n'.format(item, summary[item + '_frequency']))
        file.write('\n')

    return summary
