# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import rdkit.Chem as Chem
import torch

from dgllife.data.jtvae import get_mol, get_smiles, sanitize, copy_atom

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

MST_MAX_WEIGHT = 100
MAX_NCAND = 2000

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

# Bond type not considered because all aromatic (so SINGLE matches DOUBLE)

def ring_bond_equal(b1, b2, reverse=False):
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])

def attach_mols_nx(ctr_mol, neighbors, prev_nodes, nei_amap):
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
    ctr_mol = copy_edit_mol(ctr_mol)
    nei_amap = {nei['nid']: {} for nei in prev_nodes + neighbors}

    for nei_id, ctr_atom, nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom

    ctr_mol = attach_mols_nx(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()

# This version records idx mapping between ctr_mol and nei_mol

def enum_attach_nx(ctr_mol, nei_node, amap, singletons):
    nei_mol, nei_idx = nei_node['mol'], nei_node['nid']
    att_confs = []
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
                        new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetBeginAtom(
                        ).GetIdx()), (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetEndAtom().GetIdx())]
                        att_confs.append(new_amap)

                    if ring_bond_equal(b1, b2, reverse=True):
                        new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetEndAtom(
                        ).GetIdx()), (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetBeginAtom().GetIdx())]
                        att_confs.append(new_amap)
    return att_confs

# Try rings first: Speed-Up

def enum_assemble_nx(node, neighbors, prev_nodes=[], prev_amap=[]):
    all_attach_confs = []
    singletons = [nei_node['nid'] for nei_node in neighbors +
                  prev_nodes if nei_node['mol'].GetNumAtoms() == 1]

    def search(cur_amap, depth):
        if len(all_attach_confs) > MAX_NCAND:
            return
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return

        nei_node = neighbors[depth]
        cand_amap = enum_attach_nx(node['mol'], nei_node, cur_amap, singletons)
        cand_smiles = set()
        candidates = []
        for amap in cand_amap:
            cand_mol = local_attach_nx(
                node['mol'], neighbors[:depth+1], prev_nodes, amap)
            cand_mol = sanitize(cand_mol)
            if cand_mol is None:
                continue
            smiles = get_smiles(cand_mol)
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

# Only used for debugging purpose

def dfs_assemble_nx(graph, cur_mol, global_amap, fa_amap, cur_node_id, fa_node_id):
    cur_node = graph.nodes_dict[cur_node_id]
    fa_node = graph.nodes_dict[fa_node_id] if fa_node_id is not None else None

    fa_nid = fa_node['nid'] if fa_node is not None else -1
    prev_nodes = [fa_node] if fa_node is not None else []

    children_id = [nei for nei in graph[cur_node_id]
                   if graph.nodes_dict[nei]['nid'] != fa_nid]
    children = [graph.nodes_dict[nei] for nei in children_id]
    neighbors = [nei for nei in children if nei['mol'].GetNumAtoms() > 1]
    neighbors = sorted(
        neighbors, key=lambda x: x['mol'].GetNumAtoms(), reverse=True)
    singletons = [nei for nei in children if nei['mol'].GetNumAtoms() == 1]
    neighbors = singletons + neighbors

    cur_amap = [(fa_nid, a2, a1)
                for nid, a1, a2 in fa_amap if nid == cur_node['nid']]
    cands = enum_assemble_nx(
        graph.nodes_dict[cur_node_id], neighbors, prev_nodes, cur_amap)
    if len(cands) == 0:
        return

    cand_smiles, _, cand_amap = zip(*cands)
    label_idx = cand_smiles.index(cur_node['label'])
    label_amap = cand_amap[label_idx]

    for nei_id, ctr_atom, nei_atom in label_amap:
        if nei_id == fa_nid:
            continue
        global_amap[nei_id][nei_atom] = global_amap[cur_node['nid']][ctr_atom]

    # father is already attached
    cur_mol = attach_mols_nx(cur_mol, children, [], global_amap)
    for nei_node_id, nei_node in zip(children_id, children):
        if not nei_node['is_leaf']:
            dfs_assemble_nx(graph, cur_mol, global_amap,
                            label_amap, nei_node_id, cur_node_id)

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
