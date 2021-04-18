# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# MolTree in JTVAE

import dgl
import numpy as np
import torch

from rdkit import Chem

from .chemutils import get_mol, decode_stereo, get_clique_mol, get_smiles, enum_assemble, \
    set_atommap, tree_decomp

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
        self.nodes_dict = {}

        # Stereo Generation
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            self.smiles3D = None
            self.smiles2D = None
            self.stereo_cands = []
            self.graph = dgl.graph(([], []), idtype=torch.int32)
            return

        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)

        # Junction tree construction
        cliques, edges = tree_decomp(self.mol)
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            csmiles = get_smiles(cmol)
            self.nodes_dict[i] = {'smiles': csmiles, 'mol': get_mol(csmiles), 'clique': c}

            if min(c) == 0:
                root = i

        # Make the clique with atom ID 0 the root
        root_changed = False
        if root > 0:
            root_changed = True
            for attr in self.nodes_dict[0]:
                self.nodes_dict[0][attr], self.nodes_dict[root][attr] = \
                    self.nodes_dict[root][attr], self.nodes_dict[0][attr]

        # Construct DGLGraph for the junction tree
        src = np.zeros((len(edges) * 2,), dtype='int')
        dst = np.zeros((len(edges) * 2,), dtype='int')

        def _switch_id(nid, root, root_changed):
            if not root_changed:
                return nid
            if nid == root:
                return 0
            elif nid == 0:
                return root
            else:
                return nid

        for i, (_x, _y) in enumerate(edges):
            x = _switch_id(_x, root, root_changed)
            y = _switch_id(_y, root, root_changed)

            src[2 * i] = x
            dst[2 * i] = y
            src[2 * i + 1] = y
            dst[2 * i + 1] = x

        self.graph = dgl.graph((src, dst), num_nodes=len(cliques), idtype=torch.int32)

        for i in self.nodes_dict:
            self.nodes_dict[i]['nid'] = i + 1
            if self.graph.out_degrees(i) > 1:
                # Leaf node mol is not marked
                set_atommap(self.nodes_dict[i]['mol'], self.nodes_dict[i]['nid'])
            self.nodes_dict[i]['is_leaf'] = (self.graph.out_degrees(i) == 1)

    def size(self):
        """Get the number of nodes in the junction tree.

        Returns
        -------
        int
            Number of nodes in the junction tree.
        """
        return self.graph.num_nodes()

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

        for j in self.graph.successors(i).numpy():
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
        node['label'] = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        node['label_mol'] = get_mol(node['label'])

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return node['label']

    def recover(self):
        """Get the SMILES string corresponding to all clusters in the original molecule."""
        for i in self.nodes_dict:
            self._recover_node(i, self.mol)

    def _assemble_node(self, i):
        """Assemble a cluster with its successors.

        Parameters
        ----------
        i : int
            The id of a cluster.
        """
        neighbors = [self.nodes_dict[j] for j in self.graph.successors(i).numpy()
                     if self.nodes_dict[j]['mol'].GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x['mol'].GetNumAtoms(), reverse=True)
        singletons = [self.nodes_dict[j] for j in self.graph.successors(i).numpy()
                      if self.nodes_dict[j]['mol'].GetNumAtoms() == 1]
        # All successors sorted based on their size
        neighbors = singletons + neighbors

        cands = enum_assemble(self.nodes_dict[i], neighbors)

        if len(cands) > 0:
            cands, cand_mols, _ = list(zip(*cands))
            self.nodes_dict[i]['cands'] = list(cands)
            self.nodes_dict[i]['cand_mols'] = list(cand_mols)
        else:
            self.nodes_dict[i]['cands'] = []
            self.nodes_dict[i]['cand_mols'] = []

    def assemble(self):
        """Assemble each cluster with its successors."""
        for i in self.nodes_dict:
            self._assemble_node(i)
