# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dataset for JTVAE

import numpy as np
import torch

from collections import defaultdict
from dgl import batch, graph
from dgl.data.utils import get_download_dir, _get_dgl_url, download, extract_archive
from functools import partial
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from torch.utils.data import Dataset

from ..utils.featurizers import BaseAtomFeaturizer, BaseBondFeaturizer, ConcatFeaturizer, \
    atom_type_one_hot, atom_degree_one_hot, atom_formal_charge_one_hot, atom_chiral_tag_one_hot, \
    atom_is_aromatic, bond_type_one_hot, bond_is_in_ring, bond_stereo_one_hot
from ..utils.mol_to_graph import mol_to_bigraph

__all__ = ['DGLMolTree',
           'JTVAEDataset',
           'JTVAECollator']

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
        A deep copy of the input atom object.
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

def local_attach_nx(ctr_mol, neighbors, prev_nodes, amap_list):
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

    ctr_mol = attach_mols_nx(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()

def enum_attach_nx(ctr_mol, nei_node, amap, singletons):
    """Enumerate possible ways to attach a cluster to the central molecule.

    Parameters
    ----------
    ctr_mol : rdkit.Chem.rdchem.Mol
        The central molecule.
    nei_node : dict
        Information for a cluster to attach to the central molecule.
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
    black_list = [atom_idx for nei_id, atom_idx,
                  _ in amap if nei_id in singletons]
    ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx()
                 not in black_list]
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
                        new_amap = amap + [(nei_idx,
                                            b1.GetBeginAtom().GetIdx(),
                                            b2.GetBeginAtom().GetIdx()),
                                           (nei_idx,
                                            b1.GetEndAtom().GetIdx(),
                                            b2.GetEndAtom().GetIdx())]
                        att_confs.append(new_amap)

                    if ring_bond_equal(b1, b2, reverse=True):
                        new_amap = amap + [(nei_idx,
                                            b1.GetBeginAtom().GetIdx(),
                                            b2.GetEndAtom().GetIdx()),
                                           (nei_idx,
                                            b1.GetEndAtom().GetIdx(),
                                            b2.GetBeginAtom().GetIdx())]
                        att_confs.append(new_amap)
    return att_confs

def enum_assemble_nx(node, neighbors, prev_nodes=None, prev_amap=None):
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
    singletons = [nei_node['nid'] for nei_node in neighbors +
                  prev_nodes if nei_node['mol'].GetNumAtoms() == 1]

    def search(cur_amap, depth, max_ncand=2000):
        if len(all_attach_confs) > max_ncand:
            return
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return

        nei_node = neighbors[depth]
        # Enumerate possible ways to attach nei_node to the central molecule
        cand_amap = enum_attach_nx(node['mol'], nei_node, cur_amap, singletons)
        cand_smiles = set()
        candidates = []
        for amap in cand_amap:
            cand_mol = local_attach_nx(
                node['mol'], neighbors[:depth+1], prev_nodes, amap)
            cand_mol = sanitize(cand_mol)
            if cand_mol is None:
                continue
            smiles = Chem.MolToSmiles(cand_mol, kekuleSmiles=True)
            if smiles in cand_smiles:
                continue
            cand_smiles.add(smiles)
            candidates.append(amap)

        if len(candidates) == 0:
            return []

        for new_amap in candidates:
            search(new_amap, depth + 1)

    search(prev_amap, 0)
    cand_smiles = set()
    candidates = []
    for amap in all_attach_confs:
        cand_mol = local_attach_nx(node['mol'], neighbors, prev_nodes, amap)
        cand_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cand_mol))
        smiles = Chem.MolToSmiles(cand_mol)
        if smiles in cand_smiles:
            continue
        cand_smiles.add(smiles)
        Chem.Kekulize(cand_mol)
        candidates.append((smiles, cand_mol, amap))

    return candidates

class DGLMolTree():
    """Junction tree.

    Parameters
    ----------
    smiles : str
        A SMILES string.
    """
    def __init__(self, smiles):
        self.smiles = smiles
        self.nodes_dict = {}

        if smiles is None:
            self.g = graph(([], []), idtype=torch.int32)
            return

        self.mol = get_mol(smiles)

        # Stereo Generation
        mol = Chem.MolFromSmiles(smiles)
        self.smiles_3d = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles_2d = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles_2d)

        # Junction tree construction
        cliques, edges = tree_decomp(self.mol)
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            csmiles = Chem.MolToSmiles(cmol, kekuleSmiles=True)
            self.nodes_dict[i] = dict(
                smiles=csmiles,
                mol=get_mol(csmiles),
                clique=c,
            )
            if min(c) == 0:
                root = i

        # The clique with atom ID 0 becomes root
        if root > 0:
            for attr in self.nodes_dict[0]:
                self.nodes_dict[0][attr], self.nodes_dict[root][attr] = \
                    self.nodes_dict[root][attr], self.nodes_dict[0][attr]

        src = np.zeros((len(edges) * 2,), dtype='int')
        dst = np.zeros((len(edges) * 2,), dtype='int')
        for i, (_x, _y) in enumerate(edges):
            if _x == root:
                x = 0
            elif _x == 0:
                x = root
            else:
                x = _x

            if _y == root:
                y = 0
            elif _y == 0:
                y = root
            else:
                y = _y

            src[2 * i] = x
            dst[2 * i] = y
            src[2 * i + 1] = y
            dst[2 * i + 1] = x

        self.g = graph((src, dst), num_nodes=len(cliques), idtype=torch.int32)

        for i in self.nodes_dict:
            self.nodes_dict[i]['nid'] = i + 1
            if self.g.out_degrees(i) > 1:  # Leaf node mol is not marked
                set_atommap(self.nodes_dict[i]['mol'], self.nodes_dict[i]['nid'])
            self.nodes_dict[i]['is_leaf'] = (self.g.out_degrees(i) == 1)

    def treesize(self):
        """Get the number of nodes in the junction tree.

        Returns
        -------
        int
            Get the number of nodes (clusters) in the junction tree.
        """
        return self.g.num_nodes()

    def _recover_node(self, i, original_mol):
        """Get the SMILES string corresponding to the i-th cluster in the
        original molecule.

        Parameters
        ----------
        i : int
            The id of a cluster.
        original_mol : rdkit.Chem.rdchem.Mol
            The original molecule.

        Returns
        -------
        str
            A SMILES string.
        """
        node = self.nodes_dict[i]

        clique = []
        clique.extend(node['clique'])
        if not node['is_leaf']:
            for cidx in node['clique']:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(node['nid'])

        for j in self.g.successors(i).numpy():
            nei_node = self.nodes_dict[j]
            clique.extend(nei_node['clique'])
            if nei_node['is_leaf']:  # Leaf node, no need to mark
                continue
            for cidx in nei_node['clique']:
                # allow singleton node override the atom mapping
                if cidx not in node['clique'] or len(nei_node['clique']) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node['nid'])

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        node['label'] = Chem.MolToSmiles(Chem.MolFromSmiles(
            Chem.MolToSmiles(label_mol, kekuleSmiles=True)))
        node['label_mol'] = get_mol(node['label'])

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return node['label']

    def recover(self):
        """Get the SMILES strings corresponding to all clusters in the
        original molecule."""
        for i in self.nodes_dict:
            self._recover_node(i, self.mol)

    def _assemble_node(self, i):
        """Assemble a cluster with its successors.

        Parameters
        ----------
        i : int
            The id of a cluster.
        """
        neighbors = [self.nodes_dict[j] for j in self.g.successors(i).numpy()
                     if self.nodes_dict[j]['mol'].GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x['mol'].GetNumAtoms(), reverse=True)
        singletons = [self.nodes_dict[j] for j in self.g.successors(i).numpy()
                      if self.nodes_dict[j]['mol'].GetNumAtoms() == 1]
        # All successors sorted based on their size
        neighbors = singletons + neighbors

        cands = enum_assemble_nx(self.nodes_dict[i], neighbors)

        if len(cands) > 0:
            self.nodes_dict[i]['cands'], self.nodes_dict[i]['cand_mols'], _ = list(zip(*cands))
            self.nodes_dict[i]['cands'] = list(self.nodes_dict[i]['cands'])
            self.nodes_dict[i]['cand_mols'] = list(self.nodes_dict[i]['cand_mols'])
        else:
            self.nodes_dict[i]['cands'] = []
            self.nodes_dict[i]['cand_mols'] = []

    def assemble(self):
        """Assemble each cluster with its successors"""
        for i in self.nodes_dict:
            self._assemble_node(i)

def _set_node_id(mol_tree, vocab):
    """Get the id corresponding to each cluster in the vocabulary

    Parameters
    ----------
    mol_tree : MolTree
        A junction tree for a molecule.
    vocab : Vocab
        A vocabulary.

    Returns
    -------
    list of int
        The i-th element corresponds to the id of the i-th cluster in the vocabulary.
    """
    wid = []
    for i, node in enumerate(mol_tree.nodes_dict):
        mol_tree.nodes_dict[node]['idx'] = i
        wid.append(vocab.get_index(mol_tree.nodes_dict[node]['smiles']))

    return wid

def get_atom_featurizer_enc():
    """Get the atom featurizer for encoding.

    Returns
    -------
    BaseAtomFeaturizer
        The atom featurizer for encoding.
    """
    featurizer = BaseAtomFeaturizer({'x': ConcatFeaturizer([
        partial(atom_type_one_hot,
                allowable_set=['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                               'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn'],
                encode_unknown=True),
        partial(atom_degree_one_hot, allowable_set=[0, 1, 2, 3, 4], encode_unknown=True),
        partial(atom_formal_charge_one_hot, allowable_set=[-1, -2, 1, 2],
                encode_unknown=True),
        partial(atom_chiral_tag_one_hot,
                allowable_set=[Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                               Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                               Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW],
                encode_unknown=True),
        atom_is_aromatic
    ])})
    return featurizer

def get_bond_featurizer_enc():
    """Get the bond featurizer for encoding.

    Returns
    -------
    BaseBondFeaturizer
        The bond featurizer for encoding.
    """
    featurizer = BaseBondFeaturizer({'x': ConcatFeaturizer([
        bond_type_one_hot,
        bond_is_in_ring,
        partial(bond_stereo_one_hot,
                allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                               Chem.rdchem.BondStereo.STEREOANY,
                               Chem.rdchem.BondStereo.STEREOZ,
                               Chem.rdchem.BondStereo.STEREOE,
                               Chem.rdchem.BondStereo.STEREOCIS],
                encode_unknown=True)
    ])})
    return featurizer

def get_atom_featurizer_dec():
    """Get the atom featurizer for decoding.

    Returns
    -------
    BaseAtomFeaturizer
        The atom featurizer for decoding.
    """
    featurizer = BaseAtomFeaturizer({'x': ConcatFeaturizer([
        partial(atom_type_one_hot,
                allowable_set=['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                               'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn'],
                encode_unknown=True),
        partial(atom_degree_one_hot, allowable_set=[0, 1, 2, 3, 4], encode_unknown=True),
        partial(atom_formal_charge_one_hot, allowable_set=[-1, -2, 1, 2],
                encode_unknown=True),
        atom_is_aromatic
    ])})
    return featurizer

def get_bond_featurizer_dec():
    """Get the bond featurizer for decoding.

    Returns
    -------
    BaseBondFeaturizer
        The bond featurizer for decoding.
    """
    featurizer = BaseBondFeaturizer({'x': ConcatFeaturizer([
        bond_type_one_hot, bond_is_in_ring
    ])})
    return featurizer

def mol2dgl_enc(smiles, atom_featurizer, bond_featurizer):
    """Convert a SMILES to a DGLGraph for encoding.

    Parameters
    ----------
    smiles : str
        A SMILES string.
    atom_featurizer : callable
        Function for featurizing atoms.
    bond_featurizer : callable
        Function for featurizing bonds.

    Returns
    -------
    g : DGLGraph
        The DGLGraph for encoding. g.ndata['x'] stores the atom features and
        g.edata['x'] stores the bond features.
    """
    mol = get_mol(smiles)
    return mol_to_bigraph(mol=mol,
                          node_featurizer=atom_featurizer,
                          edge_featurizer=bond_featurizer,
                          canonical_atom_order=False)

def mol2dgl_dec(cand_batch, atom_featurizer, bond_featurizer):
    """Convert a batch of candidate molecules to DGLGraphs for decoding.

    Parameters
    ----------
    cand_batch : list of 3-tuples
        Each tuple consists of the candidate molecule, the junction tree of
        the original molecule and the id for the cluster.
    atom_featurizer : callable
        Function for featurizing atoms.
    bond_featurizer : callable
        Function for featurizing bonds.

    Returns
    -------
    g_list : list of DGLGraph
        Each graph corresponds to a candidate molecule decoded. g_list[i].ndata['x']
        stores the atom features and g_list[i].edata['x'] stores the bond features.
    1D LongTensor
        The edges in the junction tree corresponding to the edges in ``g_list``.
    1D LongTensor
        The edges in the batched ``g_list``.
    1D LongTensor
        The nodes in the batched ``g_list``.
    """
    # Note that during graph decoding they don't predict stereochemistry-related
    # characteristics (i.e. Chiral Atoms, E-Z, Cis-Trans).  Instead, they decode
    # the 2-D graph first, then enumerate all possible 3-D forms and find the
    # one with highest score.
    cand_graphs = []
    tree_mess_source_edges = []  # map these edges from trees to...
    tree_mess_target_edges = []  # these edges on candidate graphs
    tree_mess_target_nodes = []
    n_nodes = 0

    for mol, mol_tree, ctr_node_id in cand_batch:
        n_atoms = mol.GetNumAtoms()

        g = mol_to_bigraph(mol,
                           node_featurizer=atom_featurizer,
                           edge_featurizer=bond_featurizer,
                           canonical_atom_order=False)
        cand_graphs.append(g)

        if isinstance(mol_tree, DGLMolTree):
            tree_graph = mol_tree.g
        else:
            tree_graph = mol_tree

        for i, bond in enumerate(mol.GetBonds()):
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            begin_idx, end_idx = a1.GetIdx(), a2.GetIdx()
            x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
            # Tree node ID in the batch
            x_bid = mol_tree.nodes_dict[x_nid - 1]['idx'] if x_nid > 0 else -1
            y_bid = mol_tree.nodes_dict[y_nid - 1]['idx'] if y_nid > 0 else -1

            if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                if tree_graph.has_edges_between(x_bid, y_bid):
                    tree_mess_target_edges.append(
                        (begin_idx + n_nodes, end_idx + n_nodes))
                    tree_mess_source_edges.append((x_bid, y_bid))
                    tree_mess_target_nodes.append(end_idx + n_nodes)
                if tree_graph.has_edges_between(y_bid, x_bid):
                    tree_mess_target_edges.append(
                        (end_idx + n_nodes, begin_idx + n_nodes))
                    tree_mess_source_edges.append((y_bid, x_bid))
                    tree_mess_target_nodes.append(begin_idx + n_nodes)

        # Update offset
        n_nodes += n_atoms

    return cand_graphs, \
           torch.IntTensor(tree_mess_source_edges), \
           torch.IntTensor(tree_mess_target_edges), \
           torch.IntTensor(tree_mess_target_nodes)

class JTVAEDataset(Dataset):
    """Dataset for JTVAE

    JTVAE is introduced in `Junction Tree Variational Autoencoder for Molecular Graph Generation
    <https://arxiv.org/abs/1802.04364>`__.

    Parameters
    ----------
    data : str
        * If 'train' or 'test', it will use the training or test subset of the ZINC dataset.
        * Otherwise, it should be the path to a .txt file for a dataset. The .txt file should
          contain one SMILES per line.
    vocab : Vocab
        Loaded vocabulary.
    training : bool
        Whether the dataset is for training or not.
    """
    def __init__(self, data, vocab, training=True):
        dir = get_download_dir()

        _url = _get_dgl_url('dataset/jtnn.zip')
        zip_file_path = '{}/jtnn.zip'.format(dir)
        download(_url, path=zip_file_path)
        extract_archive(zip_file_path, '{}/jtnn'.format(dir))

        print('Loading data...')
        if data in ['train', 'test']:
            # ZINC subset
            data_file = '{}/jtnn/{}.txt'.format(dir, data)
        else:
            # New dataset
            data_file = data
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]
        self.vocab = vocab

        print('Loading finished')
        print('\t# samples:', len(self.data))
        self.training = training

        self.atom_featurizer_enc = get_atom_featurizer_enc()
        self.bond_featurizer_enc = get_bond_featurizer_enc()
        self.atom_featurizer_dec = get_atom_featurizer_dec()
        self.bond_featurizer_dec = get_bond_featurizer_dec()

    @staticmethod
    def move_to_device(mol_batch, device):
        """Move a data batch to the target device.

        Parameters
        ----------
        mol_batch : dict
            A batch of datapoints.
        device
            A target device.

        Returns
        -------
        dict
            The batch of datapoints moved to the target device.
        """
        trees = []
        for tr in mol_batch['mol_trees']:
            tr.g = tr.g.to(device)
            trees.append(tr)
        mol_batch['mol_trees'] = trees
        mol_batch['mol_graph_batch'] = mol_batch['mol_graph_batch'].to(device)
        if 'cand_graph_batch' in mol_batch:
            mol_batch['cand_graph_batch'] = mol_batch['cand_graph_batch'].to(device)
        if mol_batch.get('stereo_cand_graph_batch') is not None:
            mol_batch['stereo_cand_graph_batch'] = mol_batch['stereo_cand_graph_batch'].to(device)

        return mol_batch

    def __len__(self):
        """Get the size of the dataset.

        Returns
        -------
        int
            Size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Get the datapoint corresponding to the index.

        Parameters
        ----------
        idx : int
            Index for the datapoint.

        Returns
        -------
        result: dict

            The dictionary contains the following items:

            * result['mol_tree'] : MolTree
                The junction tree for the original compound.
            * result['mol_graph'] : DGLGraph
                The DGLGraph for the original compound.
            * result['wid'] : list of int
                The ids corresponding to the clusters (nodes in the junction tree)
                in the vocabulary.
            * result['cand_graphs'] : list of DGLGraph, optional
                DGLGraphs corresponding to the enumerated candidate molecules.
                This only exists when ``self.training`` is True.
            * result['tree_mess_src_e'] : 1D LongTensor, optional
                The edges in the junction tree corresponding to the edges in
                ``result['cand_graphs']``. This only exists when ``self.training`` is True.
            * result['tree_mess_tgt_e'] : 1D LongTensor, optional
                The edges in ``result['cand_graphs']``.
                This only exists when ``self.training`` is True.
            * result['tree_mess_tgt_n'] : 1D LongTensor, optional
                The nodes in ``result['cand_graphs']``.
                This only exists when ``self.training`` is True.
            * result['stereo_cand_graphs'], optional
                DGLGraphs corresponding to enumerated stereoisomers.
                This only exists when ``self.training`` is True.
            * result['stereo_cand_label'], optional
                A 2-tuple of int. The first element is the index of the ground truth
                stereoisomer in the enumerated stereoisomers. The second element is
                the number of the enumerated stereoisomers. This only exists when
                ``self.training`` is True.
        """
        smiles = self.data[idx]
        mol_tree = DGLMolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()

        wid = _set_node_id(mol_tree, self.vocab)

        # prebuild the molecule graph
        mol_graph = mol2dgl_enc(mol_tree.smiles,
                                self.atom_featurizer_enc,
                                self.bond_featurizer_enc)

        result = {
            'mol_tree': mol_tree,
            'mol_graph': mol_graph,
            'wid': wid
        }

        if not self.training:
            return result

        # prebuild the candidate graph list
        cands = []
        # Enumerate over clusters in the junction tree
        for node_id, node in mol_tree.nodes_dict.items():
            # fill in ground truth
            if node['label'] not in node['cands']:
                node['cands'].append(node['label'])
                node['cand_mols'].append(node['label_mol'])

            if node['is_leaf'] or len(node['cands']) == 1:
                continue
            cands.extend([(cand, mol_tree, node_id)
                         for cand in node['cand_mols']])
        if len(cands) > 0:
            cand_graphs, tree_mess_src_e, tree_mess_tgt_e, tree_mess_tgt_n = mol2dgl_dec(
                cands, self.atom_featurizer_dec, self.bond_featurizer_dec)
        else:
            cand_graphs = []
            tree_mess_src_e = torch.zeros(0, 2).long()
            tree_mess_tgt_e = torch.zeros(0, 2).long()
            tree_mess_tgt_n = torch.zeros(0).long()

        # prebuild the stereoisomers
        cands = mol_tree.stereo_cands
        if len(cands) > 1:
            if mol_tree.smiles_3d not in cands:
                cands.append(mol_tree.smiles_3d)

            stereo_cand_graphs = [mol2dgl_enc(c, self.atom_featurizer_enc,
                                              self.bond_featurizer_enc)
                                  for c in cands]
            stereo_cand_label = [(cands.index(mol_tree.smiles_3d), len(cands))]
        else:
            stereo_cand_graphs = []
            stereo_cand_label = []

        result.update({
            'cand_graphs': cand_graphs,
            'tree_mess_src_e': tree_mess_src_e,
            'tree_mess_tgt_e': tree_mess_tgt_e,
            'tree_mess_tgt_n': tree_mess_tgt_n,
            'stereo_cand_graphs': stereo_cand_graphs,
            'stereo_cand_label': stereo_cand_label,
            })

        return result

def _unpack_field(examples, field):
    """Get all values of examples under the specified field.

    Parameters
    ----------
    examples : iterable
        An iterable of objects.
    field : str
        The field for value retrieval.

    Returns
    -------
    list
        A list of values.
    """
    return [e[field] for e in examples]

class JTVAECollator(object):
    """Collate function for JTVAE.

    JTVAE is introduced in `Junction Tree Variational Autoencoder for Molecular Graph Generation
    <https://arxiv.org/abs/1802.04364>`__.

    Parameters
    ----------
    training : bool
        Whether the collate function is for training or not.
    """
    def __init__(self, training):
        self.training = training

    @staticmethod
    def _batch_and_set(graphs, flatten):
        """Batch graphs and set an edge feature

        Parameters
        ----------
        graphs : list of DGLGraph or list of list of DGLGraph
            Graphs to batch.
        flatten : bool
            If True, ``graphs`` is a list of list of DGLGraphs and it will
            flatten ``graphs`` before calling dgl.batch.

        Returns
        -------
        DGLGraph
            Batched DGLGraph.
        """
        if flatten:
            graphs = [g for f in graphs for g in f]
        graph_batch = batch(graphs)
        graph_batch.edata['src_x'] = torch.zeros(graph_batch.num_edges(),
                                                 graph_batch.ndata['x'].shape[1])
        return graph_batch

    def __call__(self, examples):
        """Batch multiple datapoints

        Parameters
        ----------
        examples : list of dict
            Multiple datapoints.

        Returns
        -------
        dict
            Batched datapoint.
        """
        # get list of trees
        mol_trees = _unpack_field(examples, 'mol_tree')
        wid = _unpack_field(examples, 'wid')
        for _wid, mol_tree in zip(wid, mol_trees):
            mol_tree.g.ndata['wid'] = torch.LongTensor(_wid)

        # TODO: either support pickling or get around ctypes pointers using scipy
        # batch molecule graphs
        mol_graphs = _unpack_field(examples, 'mol_graph')
        mol_graph_batch = self._batch_and_set(mol_graphs, False)

        result = {
            'mol_trees': mol_trees,
            'mol_graph_batch': mol_graph_batch
        }

        if not self.training:
            return result

        # batch candidate graphs
        cand_graphs = _unpack_field(examples, 'cand_graphs')
        cand_batch_idx = []
        tree_mess_src_e = _unpack_field(examples, 'tree_mess_src_e')
        tree_mess_tgt_e = _unpack_field(examples, 'tree_mess_tgt_e')
        tree_mess_tgt_n = _unpack_field(examples, 'tree_mess_tgt_n')

        n_graph_nodes = 0
        n_tree_nodes = 0
        for i in range(len(cand_graphs)):
            tree_mess_tgt_e[i] += n_graph_nodes
            tree_mess_src_e[i] += n_tree_nodes
            tree_mess_tgt_n[i] += n_graph_nodes
            n_graph_nodes += sum(g.num_nodes() for g in cand_graphs[i])
            n_tree_nodes += mol_trees[i].g.num_nodes()
            cand_batch_idx.extend([i] * len(cand_graphs[i]))
        tree_mess_tgt_e = torch.cat(tree_mess_tgt_e)
        tree_mess_src_e = torch.cat(tree_mess_src_e)
        tree_mess_tgt_n = torch.cat(tree_mess_tgt_n)

        cand_graph_batch = self._batch_and_set(cand_graphs, True)

        # batch stereoisomers
        stereo_cand_graphs = _unpack_field(examples, 'stereo_cand_graphs')
        stereo_cand_batch_idx = []
        for i in range(len(stereo_cand_graphs)):
            stereo_cand_batch_idx.extend([i] * len(stereo_cand_graphs[i]))

        if len(stereo_cand_batch_idx) > 0:
            stereo_cand_labels = [
                (label, length)
                for ex in _unpack_field(examples, 'stereo_cand_label')
                for label, length in ex
            ]
            stereo_cand_labels, stereo_cand_lengths = zip(*stereo_cand_labels)
            stereo_cand_graph_batch = self._batch_and_set(stereo_cand_graphs, True)
        else:
            stereo_cand_labels = []
            stereo_cand_lengths = []
            stereo_cand_graph_batch = None
            stereo_cand_batch_idx = []

        result.update({
            'cand_graph_batch': cand_graph_batch,
            'cand_batch_idx': cand_batch_idx,
            'tree_mess_tgt_e': tree_mess_tgt_e,
            'tree_mess_src_e': tree_mess_src_e,
            'tree_mess_tgt_n': tree_mess_tgt_n,
            'stereo_cand_graph_batch': stereo_cand_graph_batch,
            'stereo_cand_batch_idx': stereo_cand_batch_idx,
            'stereo_cand_labels': stereo_cand_labels,
            'stereo_cand_lengths': stereo_cand_lengths,
            })

        return result
