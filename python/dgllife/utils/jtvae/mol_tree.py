# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# MolTree in JTVAE

from rdkit import Chem

from .chemutils import get_mol, decode_stereo, get_clique_mol, get_smiles, enum_assemble, \
    set_atommap, tree_decomp

class MolTreeNode(object):
    """A node in a junction tree.

    Parameters
    ----------
    smiles : str
        A SMILES string.
    clique : list of int
        ID for the atoms corresponding to the SMILES string in the original molecule.
    """
    def __init__(self, smiles, clique=None):
        if clique is None:
            clique = []

        self.smiles = smiles
        self.mol = get_mol(self.smiles)

        self.clique = [x for x in clique]  # copy
        self.neighbors = []

    def add_neighbor(self, nei_node):
        """Add a neighbor

        Parameters
        ----------
        nei_node : MolTreeNode
            A neighboring node in the junction tree.
        """
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        """Get the SMILES string in the original molecule.

        Parameters
        ----------
        original_mol : rdkit.Chem.rdchem.Mol
            The original molecule.

        Returns
        -------
        str
            A SMILES string.
        """
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf:  # Leaf node, no need to mark
                continue
            for cidx in nei_node.clique:
                # allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        self.label_mol = get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label

    def assemble(self):
        """Assemble the node with its successors"""
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands = enum_assemble(self, neighbors)
        if len(cands) > 0:
            self.cands, self.cand_mols, _ = zip(*cands)
            self.cands = list(self.cands)
            self.cand_mols = list(self.cand_mols)
        else:
            self.cands = []
            self.cand_mols = []

class MolTree(object):
    """Junction tree.

    Parameters
    ----------
    smiles : str
        A SMILES string.
    """
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.nodes = []

        # Stereo Generation
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            self.smiles3D = None
            self.smiles2D = None
            self.stereo_cands = []
            return

        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)

        # Junction tree construction
        cliques, edges = tree_decomp(self.mol)
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0:
                root = i

        for x, y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

        for i, node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1:
                # Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        """Get the number of nodes in the junction tree.

        Returns
        -------
        int
            Number of nodes in the junction tree.
        """
        return len(self.nodes)

    def recover(self):
        """Get the SMILES string corresponding to all clusters in the original molecule."""
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        """Assemble each cluster with its successors."""
        for node in self.nodes:
            node.assemble()
