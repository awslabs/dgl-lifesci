# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Chemistry utils in JTVAE

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def get_mol(smiles):
    """Construct RDKit molecule object and kekulize it.

    Parameters
    ----------
    smiles : str
        The SMILES string for a molecule.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        RDKit molecule object for the input SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol

def decode_stereo(smiles2D):
    """Enumerate possible stereoisomers for a molecule and get corresponding 3D SMILES strings

    Parameters
    ----------
    smiles2D : str
        2D SMILES string for a molecule.

    Returns
    -------
    list of str
        3D SMILES strings for stereoisomers
    """
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))
                   for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms()
               if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D

def get_smiles(mol):
    """Get Kekule SMILES for the input molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object.

    Returns
    -------
    str
        Kekule SMILES string corresponding to the input molecule.
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

# pylint: disable=W0703
def sanitize(mol):
    """Sanitize and Kekulize the input molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        Sanitized and Kekulized RDKit molecule object.
    """
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception:
        return None
    return mol

def copy_atom(atom):
    """Make a deep copy of the input atom object.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        An RDKit atom object.

    Returns
    -------
    rdkit.Chem.rdchem.Atom
        Deep copy of the input RDKit atom object.
    """
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    """Make a deep copy of the input molecule object for editing.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object.

    Returns
    -------
    mol : rdkit.Chem.rdchem.RWMol
        Deep copy of the input RDKit molecule object for editing.
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
    """Make a deep copy of atom-induced molecule fragment.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object.
    atoms : list of int
        IDs for the atoms to induce the molecule fragment.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Sanitized atom-induced molecule fragment.
    """
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) #We assume this is not None
    return new_mol

def atom_equal(a1, a2):
    """Check if two atoms are equal in terms of symbol and formal charge.

    Parameters
    ----------
    a1 : rdkit.Chem.rdchem.Atom
        An RDKit atom object.
    a2 : rdkit.Chem.rdchem.Atom
        An RDKit atom object.

    Returns
    -------
    bool
        Whether the two atom objects are equivalent.
    """
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

def ring_bond_equal(b1, b2, reverse=False):
    """Check if two bonds are equal.

    Two bonds are equal if the their beginning and end atoms have the same symbol and
    formal charge. Bond type not considered because all aromatic (so SINGLE matches DOUBLE).

    Parameters
    ----------
    b1 : rdkit.Chem.rdchem.Bond
        An RDKit bond object.
    b2 : rdkit.Chem.rdchem.Bond
        An RDKit bond object.
    reverse : bool
        Whether to interchange the role of beginning and end atoms of the second
        bond in comparison.

    Returns
    -------
    bool
        Whether the two bonds are equal.
    """
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])

def enum_attach(ctr_mol, nei_node, amap, singletons):
    """Enumerate possible ways to attach a cluster to the central molecule.

    This version records idx mapping between ctr_mol and nei_mol.

    Parameters
    ----------
    ctr_mol : rdkit.Chem.rdchem.Mol
        The central molecule.
    nei_node : dict
        A cluster to attach to the central molecule.
    amap : list of 3-tuples
        Each tuple consists of the id of the neighboring cluster,
        the id of the atom in the central molecule and the id of the same atom in the
        neighboring cluster.
    singletons : list of int
        IDs for the neighboring singletons attached.

    Returns
    -------
    list
        Each element is of the form "amap", corresponding to an attachment configuration.
    """
    nei_mol, nei_idx = nei_node['mol'], nei_node['nid']
    att_confs = []
    # A black list of atoms in the central molecule connected to singletons
    black_list = [atom_idx for nei_id, atom_idx, _ in amap if nei_id in singletons]
    ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx() not in black_list]
    ctr_bonds = [bond for bond in ctr_mol.GetBonds()]

    if nei_mol.GetNumBonds() == 0:  # neighbor singleton
        nei_atom = nei_mol.GetAtomWithIdx(0)
        used_list = [atom_idx for _, atom_idx, _ in amap]
        for atom in ctr_atoms:
            if atom_equal(atom, nei_atom) and atom.GetIdx() not in used_list:
                new_amap = amap + [(nei_idx, atom.GetIdx(), 0)]
                att_confs.append(new_amap)

    elif nei_mol.GetNumBonds() == 1:  # neighbor is a bond
        bond = nei_mol.GetBondWithIdx(0)
        bond_val = int(bond.GetBondTypeAsDouble())
        b1, b2 = bond.GetBeginAtom(), bond.GetEndAtom()

        for atom in ctr_atoms:
            # Optimize if atom is carbon (other atoms may change valence)
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
                continue
            if atom_equal(atom, b1):
                new_amap = amap + [(nei_idx, atom.GetIdx(), b1.GetIdx())]
                att_confs.append(new_amap)
            elif atom_equal(atom, b2):
                new_amap = amap + [(nei_idx, atom.GetIdx(), b2.GetIdx())]
                att_confs.append(new_amap)
    else:
        # intersection is an atom
        for a1 in ctr_atoms:
            for a2 in nei_mol.GetAtoms():
                if atom_equal(a1, a2):
                    # Optimize if atom is carbon (other atoms may change valence)
                    if a1.GetAtomicNum() == 6 and a1.GetTotalNumHs() + a2.GetTotalNumHs() < 4:
                        continue
                    new_amap = amap + [(nei_idx, a1.GetIdx(), a2.GetIdx())]
                    att_confs.append(new_amap)

        # intersection is an bond
        if ctr_mol.GetNumBonds() > 1:
            for b1 in ctr_bonds:
                for b2 in nei_mol.GetBonds():
                    if ring_bond_equal(b1, b2):
                        new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(),
                                            b2.GetBeginAtom().GetIdx()),
                                           (nei_idx, b1.GetEndAtom().GetIdx(),
                                            b2.GetEndAtom().GetIdx())]
                        att_confs.append(new_amap)

                    if ring_bond_equal(b1, b2, reverse=True):
                        new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(),
                                            b2.GetEndAtom().GetIdx()),
                                           (nei_idx, b1.GetEndAtom().GetIdx(),
                                            b2.GetBeginAtom().GetIdx())]
                        att_confs.append(new_amap)
    return att_confs

def attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap):
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
                elif nei_id in prev_nids: #father node overrides
                    ctr_mol.RemoveBond(a1, a2)
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
    return ctr_mol

def local_attach(ctr_mol, neighbors, prev_nodes, amap_list):
    """Pre-process and attach clusters to a central molecule.

    Parameters
    ----------
    ctr_mol : rdkit.Chem.rdchem.Mol
        The central molecule.
    neighbors : list of dict
        Each element contains the information of a neighboring cluster.
    prev_nodes : list of dict
        Each element contains the information of a neighboring cluster.
    amap_list : list of 3-tuples
        Each tuple consists of the id of the neighboring cluster,
        the id of the atom in the central molecule and the id of the same atom in the
        neighboring cluster.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        The central molecule with the clusters attached.
    """
    ctr_mol = copy_edit_mol(ctr_mol)
    nei_amap = {nei['nid']: {} for nei in prev_nodes + neighbors}

    for nei_id, ctr_atom, nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom

    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()

def enum_assemble(node, neighbors, prev_nodes=None, prev_amap=None, max_ncand=2000):
    """Enumerate all possible ways to assemble the central molecule and the neighboring clusters.

    Parameters
    ----------
    node : dict
        The information of the central molecule.
    neighbors : list of dict
        Each element contains the information of a neighboring cluster.
    prev_nodes : list of dict
        Each element contains the information of a neighboring cluster.
    prev_amap : list of 3-tuples
        Each tuple consists of the id of the neighboring cluster,
        the id of the atom in the central molecule and the id of the same atom in the
        neighboring cluster.
    max_ncand : int
        Maximum number of candidate assemble ways.

    Returns
    -------
    list of 3-tuples
        Each 3-tuple consists of the SMILES, molecule and amap corresponding to a molecule
        assembled from the central molecule and the neighboring cluster. The amap is a list
        of tuples. Each tuple consists of the id of the neighboring cluster,
        the id of the atom in the central molecule and the id of the same atom in the
        neighboring cluster.
    """
    if prev_nodes is None:
        prev_nodes = []
    if prev_amap is None:
        prev_amap = []

    all_attach_confs = []
    singletons = [nei_node['nid'] for nei_node in neighbors + prev_nodes
                  if nei_node['mol'].GetNumAtoms() == 1]

    def search(cur_amap, depth):
        if len(all_attach_confs) > max_ncand:
            return
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return

        nei_node = neighbors[depth]
        cand_amap = enum_attach(node['mol'], nei_node, cur_amap, singletons)
        cand_smiles = set()
        candidates = []
        for amap in cand_amap:
            cand_mol = local_attach(node['mol'], neighbors[:depth+1], prev_nodes, amap)
            cand_mol = sanitize(cand_mol)
            if cand_mol is None:
                continue
            smiles = get_smiles(cand_mol)
            if smiles in cand_smiles:
                continue
            cand_smiles.add(smiles)
            candidates.append(amap)

        if len(candidates) == 0:
            return

        for new_amap in candidates:
            search(new_amap, depth + 1)

    search(prev_amap, 0)
    cand_smiles = set()
    candidates = []
    for amap in all_attach_confs:
        cand_mol = local_attach(node['mol'], neighbors, prev_nodes, amap)
        cand_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cand_mol))
        smiles = Chem.MolToSmiles(cand_mol)
        if smiles in cand_smiles:
            continue
        cand_smiles.add(smiles)
        Chem.Kekulize(cand_mol)
        candidates.append((smiles, cand_mol, amap))

    return candidates

def set_atommap(mol, num=0):
    """Set the atom map number for all atoms in the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule.
    num : int
        The atom map number to set for all atoms. If 0, it will clear the atom map.
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

# pylint: disable=C0200
def tree_decomp(mol, mst_max_weight=100):
    """Tree decomposition of a molecule for junction tree construction.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule.
    mst_max_weight : int
        Max weight considered in generating a minimum spanning tree.

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
    nei_list = [[] for _ in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

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
    nei_list = [[] for _ in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

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
        elif len(rings) > 2:
            # Multiple (n>2) complex rings
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

    edges = [u + (mst_max_weight - v,) for u, v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    # Compute Maximum Spanning Tree
    row, col, data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]
    return (cliques, edges)
