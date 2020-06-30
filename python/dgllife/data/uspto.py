# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# USPTO for reaction prediction

import errno
import numpy as np
import os
import random
import torch

from collections import defaultdict
from copy import deepcopy
from dgl import DGLGraph
from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive, \
    save_graphs, load_graphs
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops
from tqdm import tqdm

from ..utils.featurizers import BaseAtomFeaturizer, ConcatFeaturizer, one_hot_encoding, \
    atom_type_one_hot, atom_degree_one_hot, atom_explicit_valence_one_hot, \
    atom_implicit_valence_one_hot, atom_is_aromatic, atom_formal_charge_one_hot, \
    BaseBondFeaturizer, bond_type_one_hot, bond_is_conjugated, bond_is_in_ring
from ..utils.mol_to_graph import mol_to_bigraph, mol_to_complete_graph

__all__ = ['WLNCenterDataset',
           'USPTOCenter',
           'WLNRankDataset',
           'USPTORank']

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Atom types distinguished in featurization
atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
              'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
              'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
              'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi',
              'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs']

default_node_featurizer_center = BaseAtomFeaturizer({
    'hv': ConcatFeaturizer(
        [partial(atom_type_one_hot,
                 allowable_set=atom_types, encode_unknown=True),
         partial(atom_degree_one_hot,
                 allowable_set=list(range(5)), encode_unknown=True),
         partial(atom_explicit_valence_one_hot,
                 allowable_set=list(range(1, 6)), encode_unknown=True),
         partial(atom_implicit_valence_one_hot,
                 allowable_set=list(range(5)), encode_unknown=True),
         atom_is_aromatic]
    )
})

default_node_featurizer_rank = BaseAtomFeaturizer({
    'hv': ConcatFeaturizer(
        [partial(atom_type_one_hot,
                 allowable_set=atom_types, encode_unknown=True),
         partial(atom_formal_charge_one_hot,
                 allowable_set=[-3, -2, -1, 0, 1, 2], encode_unknown=True),
         partial(atom_degree_one_hot,
                 allowable_set=list(range(5)), encode_unknown=True),
         partial(atom_explicit_valence_one_hot,
                 allowable_set=list(range(1, 6)), encode_unknown=True),
         partial(atom_implicit_valence_one_hot,
                 allowable_set=list(range(5)), encode_unknown=True),
         atom_is_aromatic]
    )
})

default_edge_featurizer_center = BaseBondFeaturizer({
    'he': ConcatFeaturizer([
        bond_type_one_hot, bond_is_conjugated, bond_is_in_ring]
    )
})

default_edge_featurizer_rank = BaseBondFeaturizer({
    'he': ConcatFeaturizer([
        bond_type_one_hot, bond_is_in_ring]
    )
})

def default_atom_pair_featurizer(reactants):
    """Featurize each pair of atoms, which will be used in updating
    the edata of a complete DGLGraph.

    The features include the bond type between the atoms (if any) and whether
    they belong to the same molecule. It is used in the global attention mechanism.

    Parameters
    ----------
    reactants : str
        SMILES for reactants
    data_field : str
        Key for storing the features in DGLGraph.edata. Default to 'atom_pair'

    Returns
    -------
    float32 tensor of shape (V^2, 10)
        features for each pair of atoms.
    """
    # Decide the reactant membership for each atom
    atom_to_reactant = dict()
    reactant_list = reactants.split('.')
    for id, s in enumerate(reactant_list):
        mol = Chem.MolFromSmiles(s)
        for atom in mol.GetAtoms():
            atom_to_reactant[atom.GetIntProp('molAtomMapNumber') - 1] = id

    # Construct mapping from atom pair to RDKit bond object
    all_reactant_mol = Chem.MolFromSmiles(reactants)
    atom_pair_to_bond = dict()
    for bond in all_reactant_mol.GetBonds():
        atom1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
        atom2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
        atom_pair_to_bond[(atom1, atom2)] = bond
        atom_pair_to_bond[(atom2, atom1)] = bond

    def _featurize_a_bond(bond):
        return bond_type_one_hot(bond) + bond_is_conjugated(bond) + bond_is_in_ring(bond)

    features = []
    num_atoms = all_reactant_mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(num_atoms):
            pair_feature = np.zeros(10)
            if i == j:
                features.append(pair_feature)
                continue

            bond = atom_pair_to_bond.get((i, j), None)
            if bond is not None:
                pair_feature[1:7] = _featurize_a_bond(bond)
            else:
                pair_feature[0] = 1.
            pair_feature[-4] = 1. if atom_to_reactant[i] != atom_to_reactant[j] else 0.
            pair_feature[-3] = 1. if atom_to_reactant[i] == atom_to_reactant[j] else 0.
            pair_feature[-2] = 1. if len(reactant_list) == 1 else 0.
            pair_feature[-1] = 1. if len(reactant_list) > 1 else 0.
            features.append(pair_feature)

    return torch.from_numpy(np.stack(features, axis=0).astype(np.float32))

def get_pair_label(reactants_mol, graph_edits):
    """Construct labels for each pair of atoms in reaction center prediction

    Parameters
    ----------
    reactants_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for all reactants in a reaction
    graph_edits : str
        Specifying which pairs of atoms loss a bond or form a particular bond in the reaction

    Returns
    -------
    float32 tensor of shape (V^2, 5)
        Labels constructed. V for the number of atoms in the reactants.
    """
    # 0 for losing the bond
    # 1, 2, 3, 1.5 separately for forming a single, double, triple or aromatic bond.
    bond_change_to_id = {0.0: 0, 1:1, 2:2, 3:3, 1.5:4}
    pair_to_changes = defaultdict(list)
    for edit in graph_edits.split(';'):
        a1, a2, change = edit.split('-')
        atom1 = int(a1) - 1
        atom2 = int(a2) - 1
        change = bond_change_to_id[float(change)]
        pair_to_changes[(atom1, atom2)].append(change)
        pair_to_changes[(atom2, atom1)].append(change)

    num_atoms = reactants_mol.GetNumAtoms()
    labels = torch.zeros((num_atoms, num_atoms, 5))
    for pair in pair_to_changes.keys():
        i, j = pair
        labels[i, j, pair_to_changes[(j, i)]] = 1.

    return labels.reshape(-1, 5)

def get_bond_changes(reaction):
    """Get the bond changes in a reaction.

    Parameters
    ----------
    reaction : str
        SMILES for a reaction, e.g. [CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7]
        (=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5]
        [c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]. It consists of reactants,
        products and the atom mapping.

    Returns
    -------
    bond_changes : set of 3-tuples
        Each tuple consists of (atom1, atom2, change type)
        There are 5 possible values for change type. 0 for losing the bond, and 1, 2, 3, 1.5
        separately for forming a single, double, triple or aromatic bond.
    """
    reactants = Chem.MolFromSmiles(reaction.split('>')[0])
    products  = Chem.MolFromSmiles(reaction.split('>')[2])

    conserved_maps = [
        a.GetProp('molAtomMapNumber')
        for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
    bond_changes = set() # keep track of bond changes

    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps):
            continue
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()

    for bond in bonds_prev:
        if bond not in bonds_new:
            # lost bond
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], 0.0))
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                # changed bond
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))
    for bond in bonds_new:
        if bond not in bonds_prev:
            # new bond
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))

    return bond_changes

def process_line(line):
    """Process one line consisting of one reaction for working with WLN.

    Parameters
    ----------
    line : str
        One reaction in one line

    Returns
    -------
    formatted_reaction : str
        Formatted reaction
    """
    reaction = line.strip()
    bond_changes = get_bond_changes(reaction)
    formatted_reaction = '{} {}\n'.format(
        reaction, ';'.join(['{}-{}-{}'.format(x[0], x[1], x[2]) for x in bond_changes]))

    return formatted_reaction

def process_file(path, num_processes=1):
    """Pre-process a file of reactions for working with WLN.

    Parameters
    ----------
    path : str
        Path to the file of reactions
    num_processes : int
        Number of processes to use for data pre-processing. Default to 1.
    """
    with open(path, 'r') as input_file:
        lines = input_file.readlines()
    if num_processes == 1:
        results = []
        for li in lines:
            results.append(process_line(li))
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_line, lines)
    with open(path + '.proc', 'w') as output_file:
        for line in results:
            output_file.write(line)
    print('Finished processing {}'.format(path))

def load_one_reaction(line):
    """Load one reaction and check if the reactants are valid.

    Parameters
    ----------
    line : str
        One reaction and the associated graph edits

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the reactants. None will be returned if the
        reactants are not valid.
    reaction : str
        Reaction. None will be returned if the reactants are not valid.
    graph_edits : str
        Graph edits associated with the reaction. None will be returned if the
        reactants are not valid.
    """
    # Each line represents a reaction and the corresponding graph edits
    #
    # reaction example:
    # [CH3:14][OH:15].[NH2:12][NH2:13].[OH2:11].[n:1]1[n:2][cH:3][c:4]
    # ([C:7]([O:9][CH3:8])=[O:10])[cH:5][cH:6]1>>[n:1]1[n:2][cH:3][c:4]
    # ([C:7](=[O:9])[NH:12][NH2:13])[cH:5][cH:6]1
    # The reactants are on the left-hand-side of the reaction and the product
    # is on the right-hand-side of the reaction. The numbers represent atom mapping.
    #
    # graph_edits example:
    # 23-33-1.0;23-25-0.0
    # For a triplet a-b-c, a and b are the atoms that form or loss the bond.
    # c specifies the particular change, 0.0 for losing a bond, 1.0, 2.0, 3.0 and
    # 1.5 separately for forming a single, double, triple or aromatic bond.
    reaction, graph_edits = line.strip("\r\n ").split()
    reactants = reaction.split('>')[0]
    mol = Chem.MolFromSmiles(reactants)
    if mol is None:
        return None, None, None

    # Reorder atoms according to the order specified in the atom map
    atom_map_order = [-1 for _ in range(mol.GetNumAtoms())]
    for j in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(j)
        atom_map_order[atom.GetIntProp('molAtomMapNumber') - 1] = j
    mol = rdmolops.RenumberAtoms(mol, atom_map_order)

    return mol, reaction, graph_edits

def reaction_validity_full_check(reaction_file):
    """Tell valid reactions from invalid ones.

    Parameters
    ----------
    reaction_file : str
        Path to a file for reactions, where each line has the reaction SMILES and the
        corresponding graph edits.

    Returns
    -------
    valid_reactions : list
        Valid reactions for modeling
    invalid_reactions : list
        Invalid reactions for modeling
    """
    valid_reactions = []
    invalid_reactions = []
    with open(reaction_file, 'r') as file:
        for line in file:
            try:
                mol, reaction, graph_edits = load_one_reaction(line)
                assert mol is not None
                product_mol = Chem.MolFromSmiles(reaction.split('>')[2])
                assert product_mol is not None
                get_pair_label(mol, graph_edits)
                valid_reactions.append(line)
            except:
                invalid_reactions.append(line)

    return valid_reactions, invalid_reactions

class WLNCenterDataset(object):
    """Dataset for reaction center prediction with WLN

    Parameters
    ----------
    raw_file_path : str
        Path to the raw reaction file, where each line is the SMILES for a reaction.
        We will check if raw_file_path + '.proc' exists, where each line has the reaction
        SMILES and the corresponding graph edits. If not, we will preprocess
        the raw reaction file.
    mol_graph_path : str
        Path to save/load DGLGraphs for molecules.
    mol_to_graph: callable, str -> DGLGraph
        A function turning RDKit molecule instances into DGLGraphs.
        Default to :func:`dgllife.utils.mol_to_bigraph`.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. By default, we consider descriptors including atom type,
        atom degree, atom explicit valence, atom implicit valence, aromaticity.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. By default, we consider descriptors including bond type,
        whether bond is conjugated and whether bond is in ring.
    atom_pair_featurizer : callable, str -> dict
        Featurization for each pair of atoms in multiple reactants. The result will be
        used to update edata in the complete DGLGraphs. By default, the features include
        the bond type between the atoms (if any) and whether they belong to the same molecule.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to True.
    num_processes : int
        Number of processes to use for data pre-processing. Default to 1.
    check_reaction_validity : bool
        Whether to check the validity of reactions before data pre-processing, which
        will introduce additional overhead. Default to True.
    reaction_validity_result_prefix : str or None
        Prefix for saving results for checking validity of reactions.
        This argument only comes into effect if ``check_reaction_validity`` is True,
        in which case we will save valid reactions in
        ``reaction_validity_result_prefix + _valid_reactions.proc`` and
        invalid ones in ``reaction_validity_result_prefix + _invalid_reactions.proc``.
        Default to ``''``.
    """
    def __init__(self,
                 raw_file_path,
                 mol_graph_path,
                 mol_to_graph=mol_to_bigraph,
                 node_featurizer=default_node_featurizer_center,
                 edge_featurizer=default_edge_featurizer_center,
                 atom_pair_featurizer=default_atom_pair_featurizer,
                 load=True,
                 num_processes=1,
                 check_reaction_validity=True,
                 reaction_validity_result_prefix=''):
        super(WLNCenterDataset, self).__init__()

        self._atom_pair_featurizer = atom_pair_featurizer
        self.atom_pair_features = []
        self.atom_pair_labels = []
        # Map number of nodes to a corresponding complete graph
        self.complete_graphs = dict()

        path_to_reaction_file = raw_file_path + '.proc'
        print('Pre-processing graph edits from reaction data')
        process_file(raw_file_path, num_processes)

        if check_reaction_validity:
            print('Start checking validity of input reactions for modeling...')
            valid_reactions, invalid_reactions = \
                reaction_validity_full_check(path_to_reaction_file)
            print('# valid reactions {:d}'.format(len(valid_reactions)))
            print('# invalid reactions {:d}'.format(len(invalid_reactions)))
            path_to_valid_reactions = reaction_validity_result_prefix + \
                                      '_valid_reactions.proc'
            path_to_invalid_reactions = reaction_validity_result_prefix + \
                                        '_invalid_reactions.proc'
            with open(path_to_valid_reactions, 'w') as f:
                for line in valid_reactions:
                    f.write(line)
            with open(path_to_invalid_reactions, 'w') as f:
                for line in invalid_reactions:
                    f.write(line)
            path_to_reaction_file = path_to_valid_reactions

        import time
        t0 = time.time()
        full_mols, full_reactions, full_graph_edits = \
            self.load_reaction_data(path_to_reaction_file, num_processes)
        print('Time spent', time.time() - t0)

        if load and os.path.isfile(mol_graph_path):
            print('Loading previously saved graphs...')
            self.reactant_mol_graphs, _ = load_graphs(mol_graph_path)
        else:
            print('Constructing graphs from scratch...')
            if num_processes == 1:
                self.reactant_mol_graphs = []
                for mol in full_mols:
                    self.reactant_mol_graphs.append(mol_to_graph(
                        mol, node_featurizer=node_featurizer,
                        edge_featurizer=edge_featurizer, canonical_atom_order=False))
            else:
                torch.multiprocessing.set_sharing_strategy('file_system')
                with Pool(processes=num_processes) as pool:
                    self.reactant_mol_graphs = pool.map(
                        partial(mol_to_graph, node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer, canonical_atom_order=False),
                        full_mols)

            save_graphs(mol_graph_path, self.reactant_mol_graphs)

        self.mols = full_mols
        self.reactions = full_reactions
        self.graph_edits = full_graph_edits
        self.atom_pair_features.extend([None for _ in range(len(self.mols))])
        self.atom_pair_labels.extend([None for _ in range(len(self.mols))])

    def load_reaction_data(self, file_path, num_processes):
        """Load reaction data from the raw file.

        Parameters
        ----------
        file_path : str
            Path to read the file.
        num_processes : int
            Number of processes to use for data pre-processing.

        Returns
        -------
        all_mols : list of rdkit.Chem.rdchem.Mol
            RDKit molecule instances
        all_reactions : list of str
            Reactions
        all_graph_edits : list of str
            Graph edits in the reactions.
        """
        all_mols = []
        all_reactions = []
        all_graph_edits = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if num_processes == 1:
            results = []
            for li in lines:
                mol, reaction, graph_edits = load_one_reaction(li)
                results.append((mol, reaction, graph_edits))
        else:
            with Pool(processes=num_processes) as pool:
                results = pool.map(load_one_reaction, lines)

        for mol, reaction, graph_edits in results:
            if mol is None:
                continue
            all_mols.append(mol)
            all_reactions.append(reaction)
            all_graph_edits.append(graph_edits)

        return all_mols, all_reactions, all_graph_edits

    def __len__(self):
        """Get the size for the dataset.

        Returns
        -------
        int
            Number of reactions in the dataset.
        """
        return len(self.mols)

    def __getitem__(self, item):
        """Get the i-th datapoint.

        Returns
        -------
        str
            Reaction
        str
            Graph edits for the reaction
        DGLGraph
            DGLGraph for the ith molecular graph
        DGLGraph
            Complete DGLGraph, which will be needed for predicting
            scores between each pair of atoms
        float32 tensor of shape (V^2, 10)
            Features for each pair of atoms.
        float32 tensor of shape (V^2, 5)
            Labels for reaction center prediction.
            V for the number of atoms in the reactants.
        """
        mol = self.mols[item]
        num_atoms = mol.GetNumAtoms()

        if num_atoms not in self.complete_graphs:
            self.complete_graphs[num_atoms] = mol_to_complete_graph(
                mol, add_self_loop=True, canonical_atom_order=False)

        if self.atom_pair_features[item] is None:
            reactants = self.reactions[item].split('>')[0]
            self.atom_pair_features[item] = self._atom_pair_featurizer(reactants)

        if self.atom_pair_labels[item] is None:
            self.atom_pair_labels[item] = get_pair_label(mol, self.graph_edits[item])

        return self.reactions[item], self.graph_edits[item], \
               self.reactant_mol_graphs[item], \
               self.complete_graphs[num_atoms], \
               self.atom_pair_features[item], \
               self.atom_pair_labels[item]

class USPTOCenter(WLNCenterDataset):
    """USPTO dataset for reaction center prediction.

    The dataset contains reactions from patents granted by United States Patent
    and Trademark Office (USPTO), collected by Lowe [1]. Jin et al. removes duplicates
    and erroneous reactions, obtaining a set of 480K reactions. They divide it
    into 400K, 40K, and 40K for training, validation and test.

    References:

        * [1] Patent reaction extraction
        * [2] Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

    Parameters
    ----------
    subset : str
        Whether to use the training/validation/test set as in Jin et al.

        * 'train' for the training set
        * 'val' for the validation set
        * 'test' for the test set
    mol_to_graph: callable, str -> DGLGraph
        A function turning RDKit molecule instances into DGLGraphs.
        Default to :func:`dgllife.utils.mol_to_bigraph`.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. By default, we consider descriptors including atom type,
        atom degree, atom explicit valence, atom implicit valence, aromaticity.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. By default, we consider descriptors including bond type,
        whether bond is conjugated and whether bond is in ring.
    atom_pair_featurizer : callable, str -> dict
        Featurization for each pair of atoms in multiple reactants. The result will be
        used to update edata in the complete DGLGraphs. By default, the features include
        the bond type between the atoms (if any) and whether they belong to the same molecule.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to True.
    num_processes : int
        Number of processes to use for data pre-processing. Default to 1.
    """
    def __init__(self,
                 subset,
                 mol_to_graph=mol_to_bigraph,
                 node_featurizer=default_node_featurizer_center,
                 edge_featurizer=default_edge_featurizer_center,
                 atom_pair_featurizer=default_atom_pair_featurizer,
                 load=True,
                 num_processes=1):
        assert subset in ['train', 'val', 'test'], \
            'Expect subset to be "train" or "val" or "test", got {}'.format(subset)
        print('Preparing {} subset of USPTO for reaction center prediction.'.format(subset))
        self._subset = subset
        if subset == 'val':
            subset = 'valid'

        self._url = 'dataset/uspto.zip'
        data_path = get_download_dir() + '/uspto.zip'
        extracted_data_path = get_download_dir() + '/uspto'
        download(_get_dgl_url(self._url), path=data_path)
        extract_archive(data_path, extracted_data_path)

        super(USPTOCenter, self).__init__(
            raw_file_path=extracted_data_path + '/{}.txt'.format(subset),
            mol_graph_path=extracted_data_path + '/{}_mol_graphs.bin'.format(subset),
            mol_to_graph=mol_to_graph,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            atom_pair_featurizer=atom_pair_featurizer,
            load=load,
            num_processes=num_processes,
            check_reaction_validity=False)

    @property
    def subset(self):
        """Get the subset used for USPTOCenter

        Returns
        -------
        str

            * 'full' for the complete dataset
            * 'train' for the training set
            * 'val' for the validation set
            * 'test' for the test set
        """
        return self._subset

def mkdir_p(path):
    """Create a folder for the given path.

    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_one_reaction_rank(line):
    """Load one reaction and check if the reactants are valid.

    Parameters
    ----------
    line : str
        One reaction and the associated graph edits

    Returns
    -------
    reactants_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the reactants. None will be returned if the
        line is not valid.
    product_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the product. None will be returned if the line is not valid.
    reaction_real_bond_changes : list of 3-tuples
        Real bond changes in the reaction. Each tuple is of form (atom1, atom2, change_type). For
        change_type, 0.0 stands for losing a bond, 1.0, 2.0, 3.0 and 1.5 separately stands for
        forming a single, double, triple or aromatic bond.
    """
    # Each line represents a reaction and the corresponding graph edits
    #
    # reaction example:
    # [CH3:14][OH:15].[NH2:12][NH2:13].[OH2:11].[n:1]1[n:2][cH:3][c:4]
    # ([C:7]([O:9][CH3:8])=[O:10])[cH:5][cH:6]1>>[n:1]1[n:2][cH:3][c:4]
    # ([C:7](=[O:9])[NH:12][NH2:13])[cH:5][cH:6]1
    # The reactants are on the left-hand-side of the reaction and the product
    # is on the right-hand-side of the reaction. The numbers represent atom mapping.
    #
    # graph_edits example:
    # 23-33-1.0;23-25-0.0
    # For a triplet a-b-c, a and b are the atoms that form or loss the bond.
    # c specifies the particular change, 0.0 for losing a bond, 1.0, 2.0, 3.0 and
    # 1.5 separately for forming a single, double, triple or aromatic bond.
    reaction, graph_edits = line.strip("\r\n ").split()
    reactants, _, product = reaction.split('>')
    reactants_mol = Chem.MolFromSmiles(reactants)
    if reactants_mol is None:
        return None, None, None, None, None

    product_mol = Chem.MolFromSmiles(product)
    if product_mol is None:
        return None, None, None, None, None

    # Reorder atoms according to the order specified in the atom map
    atom_map_order = [-1 for _ in range(reactants_mol.GetNumAtoms())]
    for j in range(reactants_mol.GetNumAtoms()):
        atom = reactants_mol.GetAtomWithIdx(j)
        atom_map_order[atom.GetIntProp('molAtomMapNumber') - 1] = j
    reactants_mol = rdmolops.RenumberAtoms(reactants_mol, atom_map_order)

    reaction_real_bond_changes = []
    for changed_bond in graph_edits.split(';'):
        atom1, atom2, change_type = changed_bond.split('-')
        atom1, atom2 = int(atom1) - 1, int(atom2) - 1
        reaction_real_bond_changes.append(
            (min(atom1, atom2), max(atom1, atom2), float(change_type)))

    return reactants_mol, product_mol, reaction_real_bond_changes

def load_candidate_bond_changes_for_one_reaction(line):
    """Load candidate bond changes for a reaction

    Parameters
    ----------
    line : str
        Candidate bond changes separated by ;. Each candidate bond change takes the
        form of atom1, atom2, change_type and change_score.

    Returns
    -------
    list of 4-tuples
        Loaded candidate bond changes.
    """
    reaction_candidate_bond_changes = []
    elements = line.strip().split(';')[:-1]
    for candidate in elements:
        atom1, atom2, change_type, score = candidate.split(' ')
        atom1, atom2 = int(atom1) - 1, int(atom2) - 1
        reaction_candidate_bond_changes.append((
            min(atom1, atom2), max(atom1, atom2), float(change_type), float(score)))

    return reaction_candidate_bond_changes

def bookkeep_reactant(mol, candidate_pairs):
    """Bookkeep reaction-related information of reactants.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for reactants.
    candidate_pairs : list of 2-tuples
        Pairs of atoms that ranked high by a model for reaction center prediction.
        By assumption, the two atoms are different and the first atom has a smaller
        index than the second.

    Returns
    -------
    info : dict
        Reaction-related information of reactants
    """
    num_atoms = mol.GetNumAtoms()
    info = {
        # free valence of atoms
        'free_val': [0 for _ in range(num_atoms)],
        # Whether it is a carbon atom
        'is_c': [False for _ in range(num_atoms)],
        # Whether it is a carbon atom connected to a nitrogen atom in pyridine
        'is_c2_of_pyridine': [False for _ in range(num_atoms)],
        # Whether it is a phosphorous atom
        'is_p': [False for _ in range(num_atoms)],
        # Whether it is a sulfur atom
        'is_s': [False for _ in range(num_atoms)],
        # Whether it is an oxygen atom
        'is_o': [False for _ in range(num_atoms)],
        # Whether it is a nitrogen atom
        'is_n': [False for _ in range(num_atoms)],
        'pair_to_bond_val': dict(),
        'ring_bonds': set()
    }

    # bookkeep atoms
    for j, atom in enumerate(mol.GetAtoms()):
        info['free_val'][j] += atom.GetTotalNumHs() + abs(atom.GetFormalCharge())
        # An aromatic carbon atom next to an aromatic nitrogen atom can get a
        # carbonyl b/c of bookkeeping of hydroxypyridines
        if atom.GetSymbol() == 'C':
            info['is_c'][j] = True
            if atom.GetIsAromatic():
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == 'N' and nbr.GetDegree() == 2:
                        info['is_c2_of_pyridine'][j] = True
                        break
        # A nitrogen atom should be allowed to become positively charged
        elif atom.GetSymbol() == 'N':
            info['free_val'][j] += 1 - atom.GetFormalCharge()
            info['is_n'][j] = True
        # Phosphorous atoms can form a phosphonium
        elif atom.GetSymbol() == 'P':
            info['free_val'][j] += 1 - atom.GetFormalCharge()
            info['is_p'][j] = True
        elif atom.GetSymbol() == 'O':
            info['is_o'][j] = True
        elif atom.GetSymbol() == 'S':
            info['is_s'][j] = True

    # bookkeep bonds
    for bond in mol.GetBonds():
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom1, atom2 = min(atom1, atom2), max(atom1, atom2)
        type_val = bond.GetBondTypeAsDouble()
        info['pair_to_bond_val'][(atom1, atom2)] = type_val
        if (atom1, atom2) in candidate_pairs:
            info['free_val'][atom1] += type_val
            info['free_val'][atom2] += type_val
        if bond.IsInRing():
            info['ring_bonds'].add((atom1, atom2))

    return info

def bookkeep_product(mol):
    """Bookkeep reaction-related information of atoms/bonds in products

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for products.

    Returns
    -------
    info : dict
        Reaction-related information of atoms/bonds in products
    """
    info = {
        'atoms': set()
    }
    for atom in mol.GetAtoms():
        info['atoms'].add(atom.GetAtomMapNum() - 1)

    return info

def is_connected_change_combo(combo_ids, cand_change_adj):
    """Check whether the combo of bond changes yields a connected component.

    Parameters
    ----------
    combo_ids : tuple of int
        Ids for bond changes in the combination.
    cand_change_adj : bool ndarray of shape (N, N)
        Adjacency matrix for candidate bond changes. Two candidate bond
        changes are considered adjacent if they share a common atom.
        * N for the number of candidate bond changes.

    Returns
    -------
    bool
        Whether the combo of bond changes yields a connected component
    """
    if len(combo_ids) == 1:
        return True
    multi_hop_adj = np.linalg.matrix_power(
        cand_change_adj[combo_ids, :][:, combo_ids], len(combo_ids) - 1)
    # The combo is connected if the distance between
    # any pair of bond changes is within len(combo) - 1

    return np.all(multi_hop_adj)

def is_valid_combo(combo_changes, reactant_info):
    """Whether the combo of bond changes is chemically valid.

    Parameters
    ----------
    combo_changes : list of 4-tuples
        Each tuple consists of atom1, atom2, type of bond change (in the form of related
        valence) and score for the change.
    reactant_info : dict
        Reaction-related information of reactants

    Returns
    -------
    bool
        Whether the combo of bond changes is chemically valid.
    """
    num_atoms = len(reactant_info['free_val'])
    force_even_parity = np.zeros((num_atoms,), dtype=bool)
    force_odd_parity = np.zeros((num_atoms,), dtype=bool)
    pair_seen = defaultdict(bool)
    free_val_tmp = reactant_info['free_val'].copy()
    for (atom1, atom2, change_type, score) in combo_changes:
        if pair_seen[(atom1, atom2)]:
            # A pair of atoms cannot have two types of changes. Even if we
            # randomly pick one, that will be reduced to a combo of less changes
            return False
        pair_seen[(atom1, atom2)] = True

        # Special valence rules
        atom1_type_val = atom2_type_val = change_type
        if change_type == 2:
            # to form a double bond
            if reactant_info['is_o'][atom1]:
                if reactant_info['is_c2_of_pyridine'][atom2]:
                    atom2_type_val = 1.
                elif reactant_info['is_p'][atom2]:
                    # don't count information of =o toward valence
                    # but require odd valence parity
                    atom2_type_val = 0.
                    force_odd_parity[atom2] = True
                elif reactant_info['is_s'][atom2]:
                    atom2_type_val = 0.
                    force_even_parity[atom2] = True
            elif reactant_info['is_o'][atom2]:
                if reactant_info['is_c2_of_pyridine'][atom1]:
                    atom1_type_val = 1.
                elif reactant_info['is_p'][atom1]:
                    atom1_type_val = 0.
                    force_odd_parity[atom1] = True
                elif reactant_info['is_s'][atom1]:
                    atom1_type_val = 0.
                    force_even_parity[atom1] = True
            elif reactant_info['is_n'][atom1] and reactant_info['is_p'][atom2]:
                atom2_type_val = 0.
                force_odd_parity[atom2] = True
            elif reactant_info['is_n'][atom2] and reactant_info['is_p'][atom1]:
                atom1_type_val = 0.
                force_odd_parity[atom1] = True
            elif reactant_info['is_p'][atom1] and reactant_info['is_c'][atom2]:
                atom1_type_val = 0.
                force_odd_parity[atom1] = True
            elif reactant_info['is_p'][atom2] and reactant_info['is_c'][atom1]:
                atom2_type_val = 0.
                force_odd_parity[atom2] = True

        reactant_pair_val = reactant_info['pair_to_bond_val'].get((atom1, atom2), None)
        if reactant_pair_val is not None:
            free_val_tmp[atom1] += reactant_pair_val - atom1_type_val
            free_val_tmp[atom2] += reactant_pair_val - atom2_type_val
        else:
            free_val_tmp[atom1] -= atom1_type_val
            free_val_tmp[atom2] -= atom2_type_val

    free_val_tmp = np.array(free_val_tmp)
    # False if 1) too many connections 2) sulfur valence not even
    # 3) phosphorous valence not odd
    if any(free_val_tmp < 0) or \
            any(aval % 2 != 0 for aval in free_val_tmp[force_even_parity]) or \
            any(aval % 2 != 1 for aval in free_val_tmp[force_odd_parity]):
        return False
    return True

def edit_mol(reactant_mols, edits, product_info):
    """Simulate reaction via graph editing

    Parameters
    ----------
    reactant_mols : rdkit.Chem.rdchem.Mol
        RDKit molecule instances for reactants.
    edits : list of 4-tuples
        Bond changes for getting the product out of the reactants in a reaction.
        Each 4-tuple is of form (atom1, atom2, change_type, score), where atom1
        and atom2 are the end atoms to form or lose a bond, change_type is the
        type of bond change and score represents the confidence for the bond change
        by a model.
    product_info : dict
        proeduct_info['atoms'] gives a set of atom ids in the ground truth product molecule.

    Returns
    -------
    str
        SMILES for the main products
    """
    bond_change_to_type = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE,
                           3: Chem.rdchem.BondType.TRIPLE, 1.5: Chem.rdchem.BondType.AROMATIC}

    new_mol = Chem.RWMol(reactant_mols)
    [atom.SetNumExplicitHs(0) for atom in new_mol.GetAtoms()]

    for atom1, atom2, change_type, score in edits:
        bond = new_mol.GetBondBetweenAtoms(atom1, atom2)
        if bond is not None:
            new_mol.RemoveBond(atom1, atom2)
        if change_type > 0:
            new_mol.AddBond(atom1, atom2, bond_change_to_type[change_type])

    pred_mol = new_mol.GetMol()
    pred_smiles = Chem.MolToSmiles(pred_mol)
    pred_list = pred_smiles.split('.')
    pred_mols = []
    for pred_smiles in pred_list:
        mol = Chem.MolFromSmiles(pred_smiles)
        if mol is None:
            continue
        atom_set = set([atom.GetAtomMapNum() - 1 for atom in mol.GetAtoms()])
        if len(atom_set & product_info['atoms']) == 0:
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        pred_mols.append(mol)

    return '.'.join(sorted([Chem.MolToSmiles(mol) for mol in pred_mols]))

def get_product_smiles(reactant_mols, edits, product_info):
    """Get the product smiles of the reaction

    Parameters
    ----------
    reactant_mols : rdkit.Chem.rdchem.Mol
        RDKit molecule instances for reactants.
    edits : list of 4-tuples
        Bond changes for getting the product out of the reactants in a reaction.
        Each 4-tuple is of form (atom1, atom2, change_type, score), where atom1
        and atom2 are the end atoms to form or lose a bond, change_type is the
        type of bond change and score represents the confidence for the bond change
        by a model.
    product_info : dict
        proeduct_info['atoms'] gives a set of atom ids in the ground truth product molecule.

    Returns
    -------
    str
        SMILES for the main products
    """
    smiles = edit_mol(reactant_mols, edits, product_info)
    if len(smiles) != 0:
        return smiles
    try:
        Chem.Kekulize(reactant_mols)
    except Exception as e:
        return smiles
    return edit_mol(reactant_mols, edits, product_info)

def generate_valid_candidate_combos():
    return NotImplementedError

def pre_process_one_reaction(info, num_candidate_bond_changes, max_num_bond_changes,
                             max_num_change_combos, mode):
    """Pre-process one reaction for candidate ranking.

    Parameters
    ----------
    info : 4-tuple
        * candidate_bond_changes : list of tuples
            The candidate bond changes for the reaction
        * real_bond_changes : list of tuples
            The real bond changes for the reaction
        * reactant_mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for reactants
        * product_mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for product
    num_candidate_bond_changes : int
        Number of candidate bond changes to consider for the ground truth reaction.
    max_num_bond_changes : int
        Maximum number of bond changes per reaction.
    max_num_change_combos : int
        Number of bond change combos to consider for each reaction.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    mode : str
        Whether the dataset is to be used for training, validation or test.

    Returns
    -------
    valid_candidate_combos : list
        valid_candidate_combos[i] gives a list of tuples, which is the i-th valid combo
        of candidate bond changes for the reaction.
    candidate_bond_changes : list of 4-tuples
        Refined candidate bond changes considered for combos.
    reactant_info : dict
        Reaction-related information of reactants.
    """
    assert mode in ['train', 'val', 'test'], \
        "Expect mode to be 'train' or 'val' or 'test', got {}".format(mode)
    candidate_bond_changes_, real_bond_changes, reactant_mol, product_mol = info
    candidate_pairs = [(atom1, atom2) for (atom1, atom2, _, _)
                       in candidate_bond_changes_]
    reactant_info = bookkeep_reactant(reactant_mol, candidate_pairs)
    if mode == 'train':
        product_info = bookkeep_product(product_mol)

    # Filter out candidate new bonds already in reactants
    candidate_bond_changes = []
    count = 0
    for (atom1, atom2, change_type, score) in candidate_bond_changes_:
        if ((atom1, atom2) not in reactant_info['pair_to_bond_val']) or \
                (reactant_info['pair_to_bond_val'][(atom1, atom2)] != change_type):
            candidate_bond_changes.append((atom1, atom2, change_type, score))
            count += 1
            if count == num_candidate_bond_changes:
                break

    # Check if two bond changes have atom in common
    cand_change_adj = np.eye(len(candidate_bond_changes), dtype=bool)
    for i in range(len(candidate_bond_changes)):
        atom1_1, atom1_2, _, _ = candidate_bond_changes[i]
        for j in range(i + 1, len(candidate_bond_changes)):
            atom2_1, atom2_2, _, _ = candidate_bond_changes[j]
            if atom1_1 == atom2_1 or atom1_1 == atom2_2 or \
                    atom1_2 == atom2_1 or atom1_2 == atom2_2:
                cand_change_adj[i, j] = cand_change_adj[j, i] = True

    # Enumerate combinations of k candidate bond changes and record
    # those that are connected and chemically valid
    valid_candidate_combos = []
    cand_change_ids = range(len(candidate_bond_changes))
    for k in range(1, max_num_bond_changes + 1):
        for combo_ids in combinations(cand_change_ids, k):
            # Check if the changed bonds form a connected component
            if not is_connected_change_combo(combo_ids, cand_change_adj):
                continue
            combo_changes = [candidate_bond_changes[j] for j in combo_ids]
            # Check if the combo is chemically valid
            if is_valid_combo(combo_changes, reactant_info):
                valid_candidate_combos.append(combo_changes)

    if mode == 'train':
        random.shuffle(valid_candidate_combos)
        # Index for the combo of candidate bond changes
        # that is equivalent to the gold combo
        real_combo_id = -1
        for j, combo_changes in enumerate(valid_candidate_combos):
            if set([(atom1, atom2, change_type) for
                    (atom1, atom2, change_type, score) in combo_changes]) == \
                    set(real_bond_changes):
                real_combo_id = j
                break

        # If we fail to find the real combo, make it the first entry
        if real_combo_id == -1:
            valid_candidate_combos = \
                [[(atom1, atom2, change_type, 0.0)
                  for (atom1, atom2, change_type) in real_bond_changes]] + \
                valid_candidate_combos
        else:
            valid_candidate_combos[0], valid_candidate_combos[real_combo_id] = \
                valid_candidate_combos[real_combo_id], valid_candidate_combos[0]

        product_smiles = get_product_smiles(
            reactant_mol, valid_candidate_combos[0], product_info)
        if len(product_smiles) > 0:
            # Remove combos yielding duplicate products
            product_smiles = set([product_smiles])
            new_candidate_combos = [valid_candidate_combos[0]]

            count = 0
            for combo in valid_candidate_combos[1:]:
                smiles = get_product_smiles(reactant_mol, combo, product_info)
                if smiles in product_smiles or len(smiles) == 0:
                    continue
                product_smiles.add(smiles)
                new_candidate_combos.append(combo)
                count += 1
                if count == max_num_change_combos:
                    break
            valid_candidate_combos = new_candidate_combos
    valid_candidate_combos = valid_candidate_combos[:max_num_change_combos]

    return valid_candidate_combos, candidate_bond_changes, reactant_info

def featurize_nodes_and_compute_combo_scores(
        node_featurizer, reactant_mol, valid_candidate_combos):
    """Featurize atoms in reactants and compute scores for combos of bond changes

    Parameters
    ----------
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    reactant_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for reactants in a reaction
    valid_candidate_combos : list
        valid_candidate_combos[i] gives a list of tuples, which is the i-th valid combo
        of candidate bond changes for the reaction.

    Returns
    -------
    node_feats : float32 tensor of shape (N, M)
        Node features for reactants, N for the number of nodes and M for the feature size
    combo_bias : float32 tensor of shape (B, 1)
        Scores for combos of bond changes, B equals len(valid_candidate_combos)
    """
    node_feats = node_featurizer(reactant_mol)['hv']
    combo_bias = torch.zeros(len(valid_candidate_combos), 1).float()
    for combo_id, combo in enumerate(valid_candidate_combos):
        combo_bias[combo_id] = sum([
            score for (atom1, atom2, change_type, score) in combo])

    return node_feats, combo_bias

def construct_graphs_rank(info, edge_featurizer):
    """Construct graphs for reactants and candidate products in a reaction and featurize
    their edges

    Parameters
    ----------
    info : 4-tuple
        * reactant_mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for reactants in a reaction
        * candidate_combos : list
            candidate_combos[i] gives a list of tuples, which is the i-th valid combo
            of candidate bond changes for the reaction.
        * candidate_bond_changes : list of 4-tuples
            Refined candidate bond changes considered for candidate products
        * reactant_info : dict
            Reaction-related information of reactants.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph.

    Returns
    -------
    reaction_graphs : list of DGLGraphs
        DGLGraphs for reactants and candidate products with edge features in edata['he'],
        where the first graph is for reactants.
    """
    reactant_mol, candidate_combos, candidate_bond_changes, reactant_info = info
    # Graphs for reactants and candidate products
    reaction_graphs = []

    # Get graph for the reactants
    reactant_graph = mol_to_bigraph(reactant_mol, edge_featurizer=edge_featurizer,
                                    canonical_atom_order=False)
    reaction_graphs.append(reactant_graph)

    candidate_bond_changes_no_score = [
        (atom1, atom2, change_type)
        for (atom1, atom2, change_type, score) in candidate_bond_changes]

    # Prepare common components across all candidate products
    breaking_reactant_neighbors = []
    common_src_list = []
    common_dst_list = []
    common_edge_feats = []
    num_bonds = reactant_mol.GetNumBonds()
    for j in range(num_bonds):
        bond = reactant_mol.GetBondWithIdx(j)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        u_sort, v_sort = min(u, v), max(u, v)
        # Whether a bond in reactants might get broken
        if (u_sort, v_sort, 0.0) not in candidate_bond_changes_no_score:
            common_src_list.extend([u, v])
            common_dst_list.extend([v, u])
            common_edge_feats.extend([reactant_graph.edata['he'][2 * j],
                                      reactant_graph.edata['he'][2 * j + 1]])
        else:
            breaking_reactant_neighbors.append((
                u_sort, v_sort, bond.GetBondTypeAsDouble()))

    for combo in candidate_combos:
        combo_src_list = deepcopy(common_src_list)
        combo_dst_list = deepcopy(common_dst_list)
        combo_edge_feats = deepcopy(common_edge_feats)
        candidate_bond_end_atoms = [
            (atom1, atom2) for (atom1, atom2, change_type, score) in combo]
        for (atom1, atom2, change_type) in breaking_reactant_neighbors:
            if (atom1, atom2) not in candidate_bond_end_atoms:
                # If a bond might be broken in some other combos but not this,
                # add it as a negative sample
                combo.append((atom1, atom2, change_type, 0.0))

        for (atom1, atom2, change_type, score) in combo:
            if change_type == 0:
                continue
            combo_src_list.extend([atom1, atom2])
            combo_dst_list.extend([atom2, atom1])
            feats = one_hot_encoding(change_type, [1.0, 2.0, 3.0, 1.5, -1])
            if (atom1, atom2) in reactant_info['ring_bonds']:
                feats[-1] = 1
            feats = torch.tensor(feats).float()
            combo_edge_feats.extend([feats, feats.clone()])

        combo_edge_feats = torch.stack(combo_edge_feats, dim=0)
        combo_graph = DGLGraph()
        combo_graph.add_nodes(reactant_graph.number_of_nodes())
        combo_graph.add_edges(combo_src_list, combo_dst_list)
        combo_graph.edata['he'] = combo_edge_feats
        reaction_graphs.append(combo_graph)

    return reaction_graphs

class WLNRankDataset(object):
    """Dataset for ranking candidate products with WLN

    Parameters
    ----------
    path_to_reaction_file : str
        Path to the processed reaction files, where each line has the reaction SMILES
        and the corresponding graph edits.
    candidate_bond_path : str
        Path to the candidate bond changes for product enumeration, where each line is
        candidate bond changes for a reaction by a WLN for reaction center prediction.
    mode : str
        'train', 'val', or 'test', indicating whether the dataset is used for training,
        validation or test.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. By default, we consider descriptors including atom type,
        atom formal charge, atom degree, atom explicit valence, atom implicit valence,
        aromaticity.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. By default, we consider descriptors including bond type
        and whether bond is in ring.
    size_cutoff : int
        By calling ``.ignore_large(True)``, we can optionally ignore reactions whose reactants
        contain more than ``size_cutoff`` atoms. Default to 100.
    max_num_changes_per_reaction : int
        Maximum number of bond changes per reaction. Default to 5.
    num_candidate_bond_changes : int
        Number of candidate bond changes to consider for each ground truth reaction.
        Default to 16.
    max_num_change_combos_per_reaction : int
        Number of bond change combos to consider for each reaction. Default to 150.
    num_processes : int
        Number of processes to use for data pre-processing. Default to 1.
    """
    def __init__(self,
                 path_to_reaction_file,
                 candidate_bond_path,
                 mode,
                 node_featurizer=default_node_featurizer_rank,
                 edge_featurizer=default_edge_featurizer_rank,
                 size_cutoff=100,
                 max_num_changes_per_reaction=5,
                 num_candidate_bond_changes=16,
                 max_num_change_combos_per_reaction=150,
                 num_processes=1):
        super(WLNRankDataset, self).__init__()

        assert mode in ['train', 'val', 'test'], \
            "Expect mode to be 'train' or 'val' or 'test', got {}".format(mode)
        self.mode = mode

        self.ignore_large_samples = False
        self.size_cutoff = size_cutoff

        self.reactant_mols, self.product_mols, self.real_bond_changes, \
        self.ids_for_small_samples = self.load_reaction_data(path_to_reaction_file, num_processes)
        self.candidate_bond_changes = self.load_candidate_bond_changes(candidate_bond_path)

        self.num_candidate_bond_changes = num_candidate_bond_changes
        self.max_num_changes_per_reaction = max_num_changes_per_reaction
        self.max_num_change_combos_per_reaction = max_num_change_combos_per_reaction
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer

    def load_reaction_data(self, file_path, num_processes):
        """Load reaction data from the raw file.

        Parameters
        ----------
        file_path : str
            Path to read the file.
        num_processes : int
            Number of processes to use for data pre-processing.

        Returns
        -------
        all_reactant_mols : list of rdkit.Chem.rdchem.Mol
            RDKit molecule instances for reactants.
        all_product_mols : list of rdkit.Chem.rdchem.Mol
            RDKit molecule instances for products if the dataset is for training and
            None otherwise.
        all_real_bond_changes : list of list
            ``all_real_bond_changes[i]`` gives a list of tuples, which are ground
            truth bond changes for a reaction.
        ids_for_small_samples : list of int
            Indices for reactions whose reactants do not contain too many atoms
        """
        print('Stage 1/2: loading reaction data...')
        all_reactant_mols = []
        all_product_mols = []
        all_real_bond_changes = []
        ids_for_small_samples = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

        def _update_from_line(id, loaded_result):
            reactants_mol, product_mol, reaction_real_bond_changes = loaded_result
            if reactants_mol is None:
                return
            all_product_mols.append(product_mol)
            all_reactant_mols.append(reactants_mol)
            all_real_bond_changes.append(reaction_real_bond_changes)
            if reactants_mol.GetNumAtoms() <= self.size_cutoff:
                ids_for_small_samples.append(id)

        if num_processes == 1:
            for id, li in enumerate(tqdm(lines)):
                loaded_line = load_one_reaction_rank(li)
                _update_from_line(id, loaded_line)
        else:
            with Pool(processes=num_processes) as pool:
                results = pool.map(
                    load_one_reaction_rank,
                    lines, chunksize=len(lines) // num_processes)
            for id in range(len(lines)):
                _update_from_line(id, results[id])

        return all_reactant_mols, all_product_mols, all_real_bond_changes, ids_for_small_samples

    def load_candidate_bond_changes(self, file_path):
        """Load candidate bond changes predicted by a WLN for reaction center prediction.

        Parameters
        ----------
        file_path : str
            Path to a file of candidate bond changes for each reaction.

        Returns
        -------
        all_candidate_bond_changes : list of list
            ``all_candidate_bond_changes[i]`` gives a list of tuples, which are candidate
            bond changes for a reaction.
        """
        print('Stage 2/2: loading candidate bond changes...')
        with open(file_path, 'r') as f:
            lines = f.readlines()

        all_candidate_bond_changes = []
        for li in tqdm(lines):
            all_candidate_bond_changes.append(
                load_candidate_bond_changes_for_one_reaction(li))

        return all_candidate_bond_changes

    def ignore_large(self, ignore=True):
        """Whether to ignore reactions where reactants contain too many atoms.

        Parameters
        ----------
        ignore : bool
            If ``ignore``, reactions where reactants contain too many atoms will be ignored.
        """
        self.ignore_large_samples = ignore

    def __len__(self):
        """Get the size for the dataset.

        Returns
        -------
        int
            Number of reactions in the dataset.
        """
        if self.ignore_large_samples:
            return len(self.ids_for_small_samples)
        else:
            return len(self.reactant_mols)

    def __getitem__(self, item):
        """Get the i-th datapoint.

        Parameters
        ----------
        item : int
            Index for the datapoint.

        Returns
        -------
        list of B + 1 DGLGraph
            The first entry in the list is the DGLGraph for the reactants and the rest are
            DGLGraphs for candidate products. Each DGLGraph has edge features in edata['he'] and
            node features in ndata['hv'].
        candidate_scores : float32 tensor of shape (B, 1)
            The sum of scores for bond changes in each combo, where B is the number of combos.
        labels : int64 tensor of shape (1, 1), optional
            Index for the true candidate product, which is always 0 with pre-processing. This is
            returned only when we are not in the training mode.
        valid_candidate_combos : list, optional
            valid_candidate_combos[i] gives a list of tuples, which is the i-th valid combo
            of candidate bond changes for the reaction. Each tuple is of form (atom1, atom2,
            change_type, score). atom1, atom2 are the atom mapping numbers - 1 of the two
            end atoms. change_type can be 0, 1, 2, 3, 1.5, separately for losing a bond, forming
            a single, double, triple, and aromatic bond.
        reactant_mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for the reactants
        real_bond_changes : list of tuples
            Ground truth bond changes in a reaction. Each tuple is of form (atom1, atom2,
            change_type). atom1, atom2 are the atom mapping numbers - 1 of the two
            end atoms. change_type can be 0, 1, 2, 3, 1.5, separately for losing a bond, forming
            a single, double, triple, and aromatic bond.
        product_mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for the product
        """
        if self.ignore_large_samples:
            item = self.ids_for_small_samples[item]

        raw_candidate_bond_changes = self.candidate_bond_changes[item]
        real_bond_changes = self.real_bond_changes[item]
        reactant_mol = self.reactant_mols[item]
        product_mol = self.product_mols[item]

        # Get valid candidate products, candidate bond changes considered and reactant info
        valid_candidate_combos, candidate_bond_changes, reactant_info = \
            pre_process_one_reaction(
                (raw_candidate_bond_changes, real_bond_changes,
                 reactant_mol, product_mol),
                self.num_candidate_bond_changes, self.max_num_changes_per_reaction,
                self.max_num_change_combos_per_reaction, self.mode)

        # Construct DGLGraphs and featurize their edges
        g_list = construct_graphs_rank(
            (reactant_mol, valid_candidate_combos,
             candidate_bond_changes, reactant_info),
            self.edge_featurizer)

        # Get node features and candidate scores
        node_feats, candidate_scores = featurize_nodes_and_compute_combo_scores(
            self.node_featurizer, reactant_mol, valid_candidate_combos)
        for g in g_list:
            g.ndata['hv'] = node_feats

        if self.mode == 'train':
            labels = torch.zeros(1, 1).long()
            return g_list, candidate_scores, labels
        else:
            reactant_mol = self.reactant_mols[item]
            real_bond_changes = self.real_bond_changes[item]
            product_mol = self.product_mols[item]
            return g_list, candidate_scores, valid_candidate_combos, \
                   reactant_mol, real_bond_changes, product_mol

class USPTORank(WLNRankDataset):
    """USPTO dataset for ranking candidate products.

    The dataset contains reactions from patents granted by United States Patent
    and Trademark Office (USPTO), collected by Lowe [1]. Jin et al. removes duplicates
    and erroneous reactions, obtaining a set of 480K reactions. They divide it
    into 400K, 40K, and 40K for training, validation and test.

    References:

        * [1] Patent reaction extraction
        * [2] Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network

    Parameters
    ----------
    subset : str
        Whether to use the training/validation/test set as in Jin et al.

        * 'train' for the training set
        * 'val' for the validation set
        * 'test' for the test set
    candidate_bond_path : str
        Path to the candidate bond changes for product enumeration, where each line is
        candidate bond changes for a reaction by a WLN for reaction center prediction.
    size_cutoff : int
        By calling ``.ignore_large(True)``, we can optionally ignore reactions whose reactants
        contain more than ``size_cutoff`` atoms. Default to 100.
    max_num_changes_per_reaction : int
        Maximum number of bond changes per reaction. Default to 5.
    num_candidate_bond_changes : int
        Number of candidate bond changes to consider for each ground truth reaction.
        Default to 16.
    max_num_change_combos_per_reaction : int
        Number of bond change combos to consider for each reaction. Default to 150.
    num_processes : int
        Number of processes to use for data pre-processing. Default to 1.
    """
    def __init__(self,
                 subset,
                 candidate_bond_path,
                 size_cutoff=100,
                 max_num_changes_per_reaction=5,
                 num_candidate_bond_changes=16,
                 max_num_change_combos_per_reaction=150,
                 num_processes=1):
        assert subset in ['train', 'val', 'test'], \
            'Expect subset to be "train" or "val" or "test", got {}'.format(subset)
        print('Preparing {} subset of USPTO for product candidate ranking.'.format(subset))
        self._subset = subset
        if subset == 'val':
            mode = 'val'
            subset = 'valid'
        else:
            mode = subset

        self._url = 'dataset/uspto.zip'
        data_path = get_download_dir() + '/uspto.zip'
        extracted_data_path = get_download_dir() + '/uspto'
        download(_get_dgl_url(self._url), path=data_path)
        extract_archive(data_path, extracted_data_path)

        super(USPTORank, self).__init__(
            path_to_reaction_file=extracted_data_path + '/{}.txt.proc'.format(subset),
            candidate_bond_path=candidate_bond_path,
            mode=mode,
            size_cutoff=size_cutoff,
            max_num_changes_per_reaction=max_num_changes_per_reaction,
            num_candidate_bond_changes=num_candidate_bond_changes,
            max_num_change_combos_per_reaction=max_num_change_combos_per_reaction,
            num_processes=num_processes)

    @property
    def subset(self):
        """Get the subset used for USPTOCenter

        Returns
        -------
        str

            * 'full' for the complete dataset
            * 'train' for the training set
            * 'val' for the validation set
            * 'test' for the test set
        """
        return self._subset
