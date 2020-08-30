# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert complexes into DGLHeteroGraphs
# pylint: disable= no-member, arguments-differ, invalid-name
import numpy as np
import dgl.backend as F

from dgl import graph, bipartite, hetero_from_relations, batch, DGLGraph

from ..utils.mol_to_graph import k_nearest_neighbors, mol_to_bigraph
from ..utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

__all__ = ['ACNN_graph_construction_and_featurization', 
           'potentialNet_graph_construction_featurization', 
           'flattern_graph']

def filter_out_hydrogens(mol):
    """Get indices for non-hydrogen atoms.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.

    Returns
    -------
    indices_left : list of int
        Indices of non-hydrogen atoms.
    """
    indices_left = []
    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        # Hydrogen atoms have an atomic number of 1.
        if atomic_num != 1:
            indices_left.append(i)
    return indices_left

def get_atomic_numbers(mol, indices):
    """Get the atomic numbers for the specified atoms.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    indices : list of int
        Specifying atoms.

    Returns
    -------
    list of int
        Atomic numbers computed.
    """
    atomic_numbers = []
    for i in indices:
        atom = mol.GetAtomWithIdx(i)
        atomic_numbers.append(atom.GetAtomicNum())
    return atomic_numbers

def flattern_graph(graph):
    """Flattern a batched graph by removing batch infomation
    """
    g = dgl.add_edges(graph, [0], [0]) # add dummy edge to new graph
    g.remove_edges(g.num_edges()-1) # remove the edge we just added
    return g

def potentialNet_graph_construction_featurization(ligand_mol,
                                              protein_mol,
                                              ligand_coordinates,
                                              protein_coordinates,
                                              max_num_ligand_atoms=None,
                                              max_num_protein_atoms=None,
                                              neighbor_cutoff=3.,
                                              max_num_neighbors=12,
                                              strip_hydrogens=False):
    """Graph construction and featurization for `PotentialNet` <link>__.
    Parameters
    ----------
    ligand_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    protein_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    ligand_coordinates : Float Tensor of shape (V1, 3)
        Atom coordinates in a ligand.
    protein_coordinates : Float Tensor of shape (V2, 3)
        Atom coordinates in a protein.
    max_num_ligand_atoms : int or None
        Maximum number of atoms in ligands for zero padding, which should be no smaller than
        ligand_mol.GetNumAtoms() if not None. If None, no zero padding will be performed.
        Default to None.
    max_num_protein_atoms : int or None
        Maximum number of atoms in proteins for zero padding, which should be no smaller than
        protein_mol.GetNumAtoms() if not None. If None, no zero padding will be performed.
        Default to None.
    neighbor_cutoff : float
        Distance cutoff to define 'neighboring'. Default to 12.
    max_num_neighbors : int
        Maximum number of neighbors allowed for each atom. Default to 12.
    strip_hydrogens : bool
        Whether to exclude hydrogen atoms. Default to False.
    """
    assert ligand_coordinates is not None, 'Expect ligand_coordinates to be provided.'
    assert protein_coordinates is not None, 'Expect protein_coordinates to be provided.'
    if max_num_ligand_atoms is not None:
        assert max_num_ligand_atoms >= ligand_mol.GetNumAtoms(), \
            'Expect max_num_ligand_atoms to be no smaller than ligand_mol.GetNumAtoms(), ' \
            'got {:d} and {:d}'.format(max_num_ligand_atoms, ligand_mol.GetNumAtoms())
    if max_num_protein_atoms is not None:
        assert max_num_protein_atoms >= protein_mol.GetNumAtoms(), \
            'Expect max_num_protein_atoms to be no smaller than protein_mol.GetNumAtoms(), ' \
            'got {:d} and {:d}'.format(max_num_protein_atoms, protein_mol.GetNumAtoms())

    if strip_hydrogens:
        # Remove hydrogen atoms and their corresponding coordinates
        ligand_atom_indices_left = filter_out_hydrogens(ligand_mol)
        protein_atom_indices_left = filter_out_hydrogens(protein_mol)
        ligand_coordinates = ligand_coordinates.take(ligand_atom_indices_left, axis=0)
        protein_coordinates = protein_coordinates.take(protein_atom_indices_left, axis=0)
    else:
        ligand_atom_indices_left = list(range(ligand_mol.GetNumAtoms()))
        protein_atom_indices_left = list(range(protein_mol.GetNumAtoms()))

    # Compute number of nodes for each type
    if max_num_ligand_atoms is None:
        num_ligand_atoms = len(ligand_atom_indices_left)
    else:
        num_ligand_atoms = max_num_ligand_atoms

    if max_num_protein_atoms is None:
        num_protein_atoms = len(protein_atom_indices_left)
    else:
        num_protein_atoms = max_num_protein_atoms

    # Construct bigraph for stage 1
    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
    ligand_bigraph = mol_to_bigraph(ligand_mol, add_self_loop=False,
                   node_featurizer=node_featurizer,
                   edge_featurizer=edge_featurizer,
                   canonical_atom_order=True,
                   explicit_hydrogens=False)
    protein_bigraph = mol_to_bigraph(protein_mol, add_self_loop=False,
                   node_featurizer=node_featurizer,
                   edge_featurizer=edge_featurizer,
                   canonical_atom_order=True,
                   explicit_hydrogens=False)

    complex_bigraph = batch([ligand_bigraph, protein_bigraph])
    # complex_bigraph.flatten() # not applicable in dgl 0.5

    # Construct knn grpah for stage 2
    complex_coordinates = np.concatenate([ligand_coordinates, protein_coordinates])
    complex_srcs, complex_dsts, complex_dists = k_nearest_neighbors(
            complex_coordinates,
            neighbor_cutoff, max_num_neighbors)
    complex_srcs = np.array(complex_srcs)
    complex_dsts = np.array(complex_dsts)
    complex_dists = np.array(complex_dists)

    complex_knn_graph = graph([])
    complex_knn_graph.add_nodes(len(complex_coordinates))
    complex_knn_graph.add_edges(complex_srcs, complex_dsts)
    
    return complex_bigraph, complex_knn_graph


# pylint: disable=C0326
def ACNN_graph_construction_and_featurization(ligand_mol,
                                              protein_mol,
                                              ligand_coordinates,
                                              protein_coordinates,
                                              max_num_ligand_atoms=None,
                                              max_num_protein_atoms=None,
                                              neighbor_cutoff=12.,
                                              max_num_neighbors=12,
                                              strip_hydrogens=False):
    """Graph construction and featurization for `Atomic Convolutional Networks for
    Predicting Protein-Ligand Binding Affinity <https://arxiv.org/abs/1703.10603>`__.

    Parameters
    ----------
    ligand_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    protein_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    ligand_coordinates : Float Tensor of shape (V1, 3)
        Atom coordinates in a ligand.
    protein_coordinates : Float Tensor of shape (V2, 3)
        Atom coordinates in a protein.
    max_num_ligand_atoms : int or None
        Maximum number of atoms in ligands for zero padding, which should be no smaller than
        ligand_mol.GetNumAtoms() if not None. If None, no zero padding will be performed.
        Default to None.
    max_num_protein_atoms : int or None
        Maximum number of atoms in proteins for zero padding, which should be no smaller than
        protein_mol.GetNumAtoms() if not None. If None, no zero padding will be performed.
        Default to None.
    neighbor_cutoff : float
        Distance cutoff to define 'neighboring'. Default to 12.
    max_num_neighbors : int
        Maximum number of neighbors allowed for each atom. Default to 12.
    strip_hydrogens : bool
        Whether to exclude hydrogen atoms. Default to False.
    """
    assert ligand_coordinates is not None, 'Expect ligand_coordinates to be provided.'
    assert protein_coordinates is not None, 'Expect protein_coordinates to be provided.'
    if max_num_ligand_atoms is not None:
        assert max_num_ligand_atoms >= ligand_mol.GetNumAtoms(), \
            'Expect max_num_ligand_atoms to be no smaller than ligand_mol.GetNumAtoms(), ' \
            'got {:d} and {:d}'.format(max_num_ligand_atoms, ligand_mol.GetNumAtoms())
    if max_num_protein_atoms is not None:
        assert max_num_protein_atoms >= protein_mol.GetNumAtoms(), \
            'Expect max_num_protein_atoms to be no smaller than protein_mol.GetNumAtoms(), ' \
            'got {:d} and {:d}'.format(max_num_protein_atoms, protein_mol.GetNumAtoms())

    if strip_hydrogens:
        # Remove hydrogen atoms and their corresponding coordinates
        ligand_atom_indices_left = filter_out_hydrogens(ligand_mol)
        protein_atom_indices_left = filter_out_hydrogens(protein_mol)
        ligand_coordinates = ligand_coordinates.take(ligand_atom_indices_left, axis=0)
        protein_coordinates = protein_coordinates.take(protein_atom_indices_left, axis=0)
    else:
        ligand_atom_indices_left = list(range(ligand_mol.GetNumAtoms()))
        protein_atom_indices_left = list(range(protein_mol.GetNumAtoms()))

    # Compute number of nodes for each type
    if max_num_ligand_atoms is None:
        num_ligand_atoms = len(ligand_atom_indices_left)
    else:
        num_ligand_atoms = max_num_ligand_atoms

    if max_num_protein_atoms is None:
        num_protein_atoms = len(protein_atom_indices_left)
    else:
        num_protein_atoms = max_num_protein_atoms

    data_dict = dict()
    num_nodes_dict = dict()

    # graph data for atoms in the ligand
    ligand_srcs, ligand_dsts, ligand_dists = k_nearest_neighbors(
        ligand_coordinates, neighbor_cutoff, max_num_neighbors)
    data_dict[('ligand_atom', 'ligand', 'ligand_atom')] = (ligand_srcs, ligand_dsts)
    num_nodes_dict['ligand_atom'] = num_ligand_atoms

    # graph data for atoms in the protein
    protein_srcs, protein_dsts, protein_dists = k_nearest_neighbors(
        protein_coordinates, neighbor_cutoff, max_num_neighbors)
    data_dict[('protein_atom', 'protein', 'protein_atom')] = (protein_srcs, protein_dsts)
    num_nodes_dict['protein_atom'] = num_protein_atoms

    # 4 graphs for complex representation, including the connection within
    # protein atoms, the connection within ligand atoms and the connection between
    # protein and ligand atoms.
    complex_srcs, complex_dsts, complex_dists = k_nearest_neighbors(
        np.concatenate([ligand_coordinates, protein_coordinates]),
        neighbor_cutoff, max_num_neighbors)
    complex_srcs = np.array(complex_srcs)
    complex_dsts = np.array(complex_dsts)
    complex_dists = np.array(complex_dists)
    offset = num_ligand_atoms

    # ('ligand_atom', 'complex', 'ligand_atom')
    inter_ligand_indices = np.intersect1d(
        (complex_srcs < offset).nonzero()[0],
        (complex_dsts < offset).nonzero()[0],
        assume_unique=True)
    data_dict[('ligand_atom', 'complex', 'ligand_atom')] = \
        (complex_srcs[inter_ligand_indices].tolist(),
         complex_dsts[inter_ligand_indices].tolist())

    # ('protein_atom', 'complex', 'protein_atom')
    inter_protein_indices = np.intersect1d(
        (complex_srcs >= offset).nonzero()[0],
        (complex_dsts >= offset).nonzero()[0],
        assume_unique=True)
    data_dict[('protein_atom', 'complex', 'protein_atom')] = \
        ((complex_srcs[inter_protein_indices] - offset).tolist(),
         (complex_dsts[inter_protein_indices] - offset).tolist())

    # ('ligand_atom', 'complex', 'protein_atom')
    ligand_protein_indices = np.intersect1d(
        (complex_srcs < offset).nonzero()[0],
        (complex_dsts >= offset).nonzero()[0],
        assume_unique=True)
    data_dict[('ligand_atom', 'complex', 'protein_atom')] = \
        (complex_srcs[ligand_protein_indices].tolist(),
         (complex_dsts[ligand_protein_indices] - offset).tolist())

    # ('protein_atom', 'complex', 'ligand_atom')
    protein_ligand_indices = np.intersect1d(
        (complex_srcs >= offset).nonzero()[0],
        (complex_dsts < offset).nonzero()[0],
        assume_unique=True)
    data_dict[('protein_atom', 'complex', 'ligand_atom')] = \
        ((complex_srcs[protein_ligand_indices] - offset).tolist(),
         complex_dsts[protein_ligand_indices].tolist())

    g = heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    g.edges['ligand'].data['distance'] = F.reshape(F.zerocopy_from_numpy(
        np.array(ligand_dists).astype(np.float32)), (-1, 1))
    g.edges['protein'].data['distance'] = F.reshape(F.zerocopy_from_numpy(
        np.array(protein_dists).astype(np.float32)), (-1, 1))
    g.edges[('ligand_atom', 'complex', 'ligand_atom')].data['distance'] = \
        F.reshape(F.zerocopy_from_numpy(
            complex_dists[inter_ligand_indices].astype(np.float32)), (-1, 1))
    g.edges[('protein_atom', 'complex', 'protein_atom')].data['distance'] = \
        F.reshape(F.zerocopy_from_numpy(
            complex_dists[inter_protein_indices].astype(np.float32)), (-1, 1))
    g.edges[('ligand_atom', 'complex', 'protein_atom')].data['distance'] = \
        F.reshape(F.zerocopy_from_numpy(
            complex_dists[ligand_protein_indices].astype(np.float32)), (-1, 1))
    g.edges[('protein_atom', 'complex', 'ligand_atom')].data['distance'] = \
        F.reshape(F.zerocopy_from_numpy(
            complex_dists[protein_ligand_indices].astype(np.float32)), (-1, 1))

    # Get atomic numbers for all atoms left and set node features
    ligand_atomic_numbers = np.array(get_atomic_numbers(ligand_mol, ligand_atom_indices_left))
    # zero padding
    ligand_atomic_numbers = np.concatenate([
        ligand_atomic_numbers, np.zeros(num_ligand_atoms - len(ligand_atom_indices_left))])
    protein_atomic_numbers = np.array(get_atomic_numbers(protein_mol, protein_atom_indices_left))
    # zero padding
    protein_atomic_numbers = np.concatenate([
        protein_atomic_numbers, np.zeros(num_protein_atoms - len(protein_atom_indices_left))])

    g.nodes['ligand_atom'].data['atomic_number'] = F.reshape(F.zerocopy_from_numpy(
        ligand_atomic_numbers.astype(np.float32)), (-1, 1))
    g.nodes['protein_atom'].data['atomic_number'] = F.reshape(F.zerocopy_from_numpy(
        protein_atomic_numbers.astype(np.float32)), (-1, 1))

    # Prepare mask indicating the existence of nodes
    ligand_masks = np.zeros((num_ligand_atoms, 1))
    ligand_masks[:len(ligand_atom_indices_left), :] = 1
    g.nodes['ligand_atom'].data['mask'] = F.zerocopy_from_numpy(
        ligand_masks.astype(np.float32))
    protein_masks = np.zeros((num_protein_atoms, 1))
    protein_masks[:len(protein_atom_indices_left), :] = 1
    g.nodes['protein_atom'].data['mask'] = F.zerocopy_from_numpy(
        protein_masks.astype(np.float32))

    return g
