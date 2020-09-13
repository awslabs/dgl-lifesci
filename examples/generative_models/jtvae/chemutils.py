# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import rdkit.Chem as Chem
import torch

from dgllife.data.jtvae import get_mol

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

# Try rings first: Speed-Up

def mol2dgl_dec(cand_batch):
    # Note that during graph decoding they don't predict stereochemistry-related
    # characteristics (i.e. Chiral Atoms, E-Z, Cis-Trans).  Instead, they decode
    # the 2-D graph first, then enumerate all possible 3-D forms and find the
    # one with highest score.
    def atom_features(atom):
        return (torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
                             + onek_encoding_unk(atom.GetDegree(),
                                                 [0, 1, 2, 3, 4, 5])
                             + onek_encoding_unk(atom.GetFormalCharge(),
                                                 [-1, -2, 1, 2, 0])
                             + [atom.GetIsAromatic()]))

    def bond_features(bond):
        bt = bond.GetBondType()
        return (torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                              bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                              bond.IsInRing()]))

    cand_graphs = []
    tree_mess_source_edges = []  # map these edges from trees to...
    tree_mess_target_edges = []  # these edges on candidate graphs
    tree_mess_target_nodes = []
    n_nodes = 0
    atom_x = []
    bond_x = []

    for mol, mol_tree, ctr_node_id in cand_batch:
        n_atoms = mol.GetNumAtoms()

        g = dgl.graph(([], []), idtype=torch.int32)

        for i, atom in enumerate(mol.GetAtoms()):
            assert i == atom.GetIdx()
            atom_x.append(atom_features(atom))
        g.add_nodes(n_atoms)

        bond_src = []
        bond_dst = []
        for i, bond in enumerate(mol.GetBonds()):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            begin_idx = a1.GetIdx()
            end_idx = a2.GetIdx()
            features = bond_features(bond)

            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_x.append(features)
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
            bond_x.append(features)

            x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
            # Tree node ID in the batch
            x_bid = mol_tree.nodes_dict[x_nid - 1]['idx'] if x_nid > 0 else -1
            y_bid = mol_tree.nodes_dict[y_nid - 1]['idx'] if y_nid > 0 else -1
            if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                if mol_tree.has_edges_between(x_bid, y_bid):
                    tree_mess_target_edges.append(
                        (begin_idx + n_nodes, end_idx + n_nodes))
                    tree_mess_source_edges.append((x_bid, y_bid))
                    tree_mess_target_nodes.append(end_idx + n_nodes)
                if mol_tree.has_edges_between(y_bid, x_bid):
                    tree_mess_target_edges.append(
                        (end_idx + n_nodes, begin_idx + n_nodes))
                    tree_mess_source_edges.append((y_bid, x_bid))
                    tree_mess_target_nodes.append(begin_idx + n_nodes)

        n_nodes += n_atoms
        g.add_edges(bond_src, bond_dst)
        cand_graphs.append(g)

    return cand_graphs, torch.stack(atom_x), \
        torch.stack(bond_x) if len(bond_x) > 0 else torch.zeros(0), \
        torch.LongTensor(tree_mess_source_edges), \
        torch.LongTensor(tree_mess_target_edges), \
        torch.LongTensor(tree_mess_target_nodes)

def mol2dgl_enc(smiles):
    def atom_features(atom):
        return (torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
                             + onek_encoding_unk(atom.GetDegree(),
                                                 [0, 1, 2, 3, 4, 5])
                             + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                             + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])
                             + [atom.GetIsAromatic()]))

    def bond_features(bond):
        bt = bond.GetBondType()
        stereo = int(bond.GetStereo())
        fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt ==
                 Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
        fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
        return (torch.Tensor(fbond + fstereo))
    n_edges = 0

    atom_x = []
    bond_x = []

    mol = get_mol(smiles)
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    graph = dgl.graph(([], []), idtype=torch.int32)
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        atom_x.append(atom_features(atom))
    graph.add_nodes(n_atoms)

    bond_src = []
    bond_dst = []
    for i, bond in enumerate(mol.GetBonds()):
        begin_idx = bond.GetBeginAtom().GetIdx()
        end_idx = bond.GetEndAtom().GetIdx()
        features = bond_features(bond)
        bond_src.append(begin_idx)
        bond_dst.append(end_idx)
        bond_x.append(features)
        # set up the reverse direction
        bond_src.append(end_idx)
        bond_dst.append(begin_idx)
        bond_x.append(features)
    graph.add_edges(bond_src, bond_dst)

    n_edges += n_bonds
    return graph, torch.stack(atom_x), \
        torch.stack(bond_x) if len(bond_x) > 0 else torch.zeros(0)
