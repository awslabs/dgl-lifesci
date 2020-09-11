# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=C0111, C0103, E1101, W0611, W0612, W0703, C0200, R1710, I1101, R1721

import rdkit.Chem as Chem

from ....data.jtvae import get_smiles, sanitize, copy_edit_mol, atom_equal, \
    ring_bond_equal, attach_mols_nx

MST_MAX_WEIGHT = 100
MAX_NCAND = 2000

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
                    # Optimize if atom is carbon (other atoms may change
                    # valence)
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

# Try rings first: Speed-Up

def enum_assemble_nx(node, neighbors, prev_nodes=None, prev_amap=None):
    if prev_nodes is None:
        prev_nodes = []
    if prev_amap is None:
        prev_amap = []
    all_attach_confs = []
    singletons = [nei_node['nid'] for nei_node in neighbors +
                  prev_nodes if nei_node['mol'].GetNumAtoms() == 1]

    def search(cur_amap, depth):
        if len(all_attach_confs) > MAX_NCAND:
            return None
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return None

        nei_node = neighbors[depth]
        cand_amap = enum_attach_nx(node['mol'], nei_node, cur_amap, singletons)
        cand_smiles = set()
        candidates = []
        for amap in cand_amap:
            cand_mol = local_attach_nx(
                node['mol'], neighbors[:depth + 1], prev_nodes, amap)
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

def dfs_assemble_nx(
        graph,
        cur_mol,
        global_amap,
        fa_amap,
        cur_node_id,
        fa_node_id):
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
