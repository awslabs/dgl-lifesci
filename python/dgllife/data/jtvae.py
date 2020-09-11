# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dataset for JTVAE

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def get_mol(smiles, kekulize=True):
    """Convert the SMILES string into an RDKit molecule object

    Parameters
    ----------
    smiles : str
        A SMILES string.
    kekulize : bool
        Whether to kekulize the molecule.

    Returns
    -------
    rdkit.Chem.rdchem.Mol or None
        A Kekulized RDKit molecule object if the input SMILES string is valid and None otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if kekulize:
        Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    """Convert an RDKit molecule object into a SMILES string.

    By default, the molecule is kekulized.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule.

    Returns
    -------
    str
        The SMILES string corresponding to the molecule.
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def decode_stereo(smiles_2d):
    """Get possible 3D SMILES by enumerating stereoisomers

    Parameters
    ----------
    smiles_2d : str
        A 2D SMILES string.

    Returns
    -------
    list of str
        List of possible 3D SMILES strings.
    """
    mol = Chem.MolFromSmiles(smiles_2d)
    # Get all possible stereoisomers for a molecule
    dec_isomers = list(EnumerateStereoisomers(mol))
    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(
        mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles_3d = [Chem.MolToSmiles(mol, isomericSmiles=True)
                for mol in dec_isomers]

    chiral_n = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms()
                if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]

    if len(chiral_n) > 0:
        for mol in dec_isomers:
            for idx in chiral_n:
                mol.GetAtomWithIdx(idx).SetChiralTag(
                    Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles_3d.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles_3d

def set_atommap(mol, num=0):
    """Set the atom map number for all atoms in the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule.
    num : int
        The atom map number to set for all atoms. If 0, it will
        clear the atom map.
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

def sanitize(mol):
    """Sanitize the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Sanitized molecule.
    """
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception:
        return None
    return mol

def copy_atom(atom):
    """Get a deep copy of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        An atom.

    Returns
    -------
    rdkit.Chem.rdchem.Atom
        A deep copy of the iinput atom object.
    """
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    """Get a deep copy of a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        A deep copy of the input molecule object.
    """
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_clique_mol(mol, atoms):
    """Get an RDKit molecule object for a fragment of the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule.
    atoms : list of int
        List of ids for the atoms in the fragment.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        The RDKit molecule object corresponding to the fragment.
    """
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    return new_mol

def get_atom_to_substructures(cliques):
    """Get the substructures that each atom belongs to.

    Parameters
    ----------
    cliques : list
        Each element is a list of int, representing a
        non-ring bond or a simple cycle.

    Returns
    -------
    nei_list : dict
        Mapping atom ids to the ids of the substructures that the atom belongs to.
    """
    nei_list = defaultdict(list)
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    return nei_list

def tree_decomp(mol, mst_max_weight=100):
    """Tree decomposition of a molecule for junction tree construction.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule.
    mst_max_weight : int
        Max weight considered in generating a minimum spanning tree

    Returns
    -------
    list
        Clusters. Each element is a list of int,
        representing the atoms that constitute the cluster.
    list
        Edges between the clusters. Each element is a 2-tuple of cluster IDs.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []

    # Find all edges not belonging to any cycles
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    # Find all simple cycles, each represented by a list of IDs of the atoms in the ring
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    # Record the non-ring bonds/simple cycles that each atom belongs to
    nei_list = get_atom_to_substructures(cliques)

    # Merge simple rings that have more than two overlapping atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2:
            continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2:
                    continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    # Remove merged simple cycles
    cliques = [c for c in cliques if len(c) > 0]
    # Record the non-ring bonds/simple cycles that each atom belongs to
    nei_list = get_atom_to_substructures(cliques)

    # Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        # In general, if len(cnei) >= 3, a singleton should be added,
        # but 1 bond + 2 ring is currently not dealt with.
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            # Add singleton clique
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
        elif len(rings) > 2:  # Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = mst_max_weight - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        # cnei[i] < cnei[j] by construction
                        edges[(c1, c2)] = len(inter)

    edges = [u + (mst_max_weight-v,) for u, v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    # Compute Maximum Spanning Tree
    row, col, data = list(zip(*edges))
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    return (cliques, edges)

def atom_equal(a1, a2):
    """Check if two atoms have the same type and formal charge.

    Parameters
    ----------
    a1 : rdkit.Chem.rdchem.Atom
        An RDKit atom object.
    a2 : rdkit.Chem.rdchem.Atom
        An RDKit atom object.

    Returns
    -------
    bool
        Whether they are equal.
    """
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

def ring_bond_equal(b1, b2, reverse=False):
    """Check whether the end atoms of the two bonds have the same type and formal charge.

    Parameters
    ----------
    b1 : rdkit.Chem.rdchem.Bond
        An RDKit bond object.
    b2 : rdkit.Chem.rdchem.Bond
        An RDKit bond object.
    reverse : bool
        Whether to reverse b2 in comparison.

    Returns
    -------
    bool
        Whether the bonds are equal.
    """
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])

def attach_mols_nx(ctr_mol, neighbors, prev_nodes, nei_amap):
    """Attach clusters to a central molecule.

    Parameters
    ----------
    ctr_mol : rdkit.Chem.rdchem.Mol
        The central molecule.
    neighbors : list of dict
        Each element contains the information of a neighboring cluster.
    prev_nodes : list of dict
        Each element contains the information of a previous cluster.
    nei_amap : dict
        nei_amap[nei_id][nei_atom] maps an atom in a neighboring cluster
        with id nei_id to an atom in the central molecule.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        The central molecule with clusters attached.
    """
    prev_nids = [node['nid'] for node in prev_nodes]
    for nei_node in prev_nodes + neighbors:
        nei_id, nei_mol = nei_node['nid'], nei_node['mol']
        amap = nei_amap[nei_id]
        for atom in nei_mol.GetAtoms():
            if atom.GetIdx() not in amap:
                new_atom = copy_atom(atom)
                amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

        if nei_mol.GetNumBonds() == 0:
            nei_atom = nei_mol.GetAtomWithIdx(0)
            ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
            ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())
        else:
            for bond in nei_mol.GetBonds():
                a1 = amap[bond.GetBeginAtom().GetIdx()]
                a2 = amap[bond.GetEndAtom().GetIdx()]
                if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
                elif nei_id in prev_nids:  # father node overrides
                    ctr_mol.RemoveBond(a1, a2)
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
    return ctr_mol
