import numpy as np
import dgl.backend as F
from dgl import DGLGraph, batch

from . import mol_to_bigraph, k_nearest_neighbors
from . import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

__all__ = ['potentialNet_graph_construction_featurization']



def potentialNet_graph_construction_featurization(ligand_mol,
                                              protein_mol,
                                              ligand_coordinates,
                                              protein_coordinates,
                                              max_num_ligand_atoms=None,
                                              max_num_protein_atoms=None,
                                              neighbor_cutoff=12.,
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
    complex_bigraph.flatten()

    # Construct knn grpah for stage 2
    complex_srcs, complex_dsts, complex_dists = k_nearest_neighbors(
            np.concatenate([ligand_coordinates, protein_coordinates]),
            neighbor_cutoff, max_num_neighbors)
    complex_srcs = np.array(complex_srcs)
    complex_dsts = np.array(complex_dsts)
    complex_dists = np.array(complex_dists)

    complex_knn_graph = DGLGraph()
    complex_knn_graph.add_nodes(num=(num_ligand_atoms + num_protein_atoms))
    complex_knn_graph.add_edges(complex_srcs, complex_dsts)

    return complex_bigraph, complex_knn_graph, ligand_bigraph