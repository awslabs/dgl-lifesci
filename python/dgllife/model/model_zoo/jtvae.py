# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# JTVAE

import copy
import dgl
import dgl.function as fn
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.traversal import bfs_edges_generator, dfs_labeled_edges_generator

from ...utils.jtvae.chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols, \
    decode_stereo, get_mol
from ...utils.jtvae.mol_tree import MolTree

__all__ = ['JTNNVAE']

MAX_NB = 8
MAX_DECODE_LEN = 100

class GRUMessage(nn.Module):
    def __init__(self, hidden_size, msg_field='m'):
        super(GRUMessage, self).__init__()

        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.msg_field = msg_field

    def forward(self, edges):
        r_1 = edges.dst['src_x_r']
        r_2 = self.U_r(edges.src['h'])
        r = torch.sigmoid(r_1 + r_2)

        return {self.msg_field: r * edges.src['h']}

class GRUUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(GRUUpdate, self).__init__()

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, node):
        z = torch.sigmoid(self.W_z(
            torch.cat([node.data['src_x'], node.data['sum_h']], dim=1)))
        h_input = torch.cat([node.data['src_x'], node.data['sum_gated_h']], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))
        new_h = (1.0 - z) * node.data['sum_h'] + z * pre_h
        return {'h': new_h}

def node_aggregate(nodes, h, embedding, W):
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim
    padding = torch.zeros(hidden_size)

    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        h_nei.extend(nei)

    h_nei = torch.cat(h_nei, dim=0).view(-1, MAX_NB, hidden_size)
    sum_h_nei = h_nei.sum(dim=1)
    x_vec = torch.LongTensor(x_idx)
    x_vec = embedding(x_vec)
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1)
    return nn.ReLU()(W(node_vec))

def level_order(forest, roots):
    device = forest.device
    edges = list(bfs_edges_generator(forest, roots))
    if len(edges) == 0:
        return []
    edges = [e.to(device) for e in edges]
    edges_back = list(bfs_edges_generator(forest, roots, reverse=True))
    edges_back = [e.to(device) for e in edges_back]
    yield from reversed(edges_back)
    yield from edges

def get_root_ids(graphs):
    # Get the ID of the root nodes, the first node of all trees
    batch_num_nodes = graphs.batch_num_nodes().cpu()
    batch_num_nodes = torch.cat([torch.tensor([0]), batch_num_nodes], dim=0)
    root_ids = torch.cumsum(batch_num_nodes, dim=0)[:-1]
    root_ids = root_ids.to(dtype=graphs.idtype)

    return root_ids

class JTNNEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(vocab.size(), hidden_size)
        else:
            self.embedding = embedding

        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gru_message = GRUMessage(hidden_size)
        self.gru_update = GRUUpdate(hidden_size)
        self.W = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, tree_graphs):
        device = tree_graphs.device
        tree_graphs = tree_graphs.local_var()
        tree_graphs.ndata['x'] = self.embedding(tree_graphs.ndata['wid'])
        tree_graphs.apply_edges(fn.copy_u('x', 'src_x'))

        line_tree_graphs = dgl.line_graph(tree_graphs, backtracking=False, shared=True)
        line_tree_graphs.ndata.update({
            'src_x_r': self.W_r(line_tree_graphs.ndata['src_x']),
            # Exploit the fact that the reduce function is a sum of incoming messages,
            # and uncomputed messages are zero vectors.
            'h': torch.zeros(line_tree_graphs.num_nodes(), self.hidden_size).to(device)
        })

        # Get the ID of the root nodes, the first node of all trees
        root_ids = get_root_ids(tree_graphs)

        for eid in level_order(tree_graphs, root_ids):
            line_tree_graphs.pull(v=eid, message_func=fn.copy_u('h', 'h_nei'),
                                  reduce_func=fn.sum('h_nei', 'sum_h'))
            line_tree_graphs.pull(v=eid, message_func=self.gru_message,
                                  reduce_func=fn.sum('m', 'sum_gated_h'))
            line_tree_graphs.apply_nodes(self.gru_update, v=eid)

        # Readout
        root_ids = root_ids.long()
        tree_graphs.edata['h'] = line_tree_graphs.ndata['h']
        tree_graphs.pull(v=root_ids, message_func=fn.copy_e('h', 'm'),
                         reduce_func=fn.sum('m', 'h'))
        root_vec = torch.cat([
            tree_graphs.ndata['x'][root_ids],
            tree_graphs.ndata['h'][root_ids]
        ], dim=1)
        root_vec = self.W(root_vec)

        return tree_graphs.edata['h'], root_vec

def dfs(stack, x, fa):
    for y in x.neighbors:
        if y.idx == fa.idx:
            continue
        stack.append((x,y,1))
        dfs(stack, y, x)
        stack.append((y,x,0))

def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i,s1 in enumerate(fa_slots):
        a1,c1,h1 = s1
        for j,s2 in enumerate(ch_slots):
            a2,c2,h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append( (i,j) )

    if len(matches) == 0: return False

    fa_match,ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True

def can_assemble(node_x, node_y):
    neis = node_x.neighbors + [node_y]
    for i,nei in enumerate(neis):
        nei.nid = i

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands = enum_assemble(node_x, neighbors)
    return len(cands) > 0

def dfs_order(forest, roots):
    edges = dfs_labeled_edges_generator(forest, roots, has_reverse_edge=True)
    for e, l in zip(*edges):
        # Exploit the fact that the reverse edge ID equals to 1 xor forward
        # edge ID. Normally, this should be done using find_edges().
        yield e ^ l, l

class JTNNDecoder(nn.Module):
    def __init__(self, vocab, hidden_size, latent_size, embedding=None):
        super(JTNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        # GRU Weights
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gru_message = GRUMessage(hidden_size)
        self.gru_update = GRUUpdate(hidden_size)

        # Feature Aggregate Weights
        self.W = nn.Linear(latent_size + hidden_size, hidden_size)
        self.U = nn.Linear(latent_size + 2 * hidden_size, hidden_size)

        # Output Weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_s = nn.Linear(hidden_size, 1)

        # Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(size_average=False)
        self.stop_loss = nn.BCEWithLogitsLoss(size_average=False)

    def get_trace(self, node):
        super_root = MolTreeNode("")
        super_root.idx = -1
        trace = []
        dfs(trace, node, super_root)
        return [(x.smiles, y.smiles, z) for x, y, z in trace]

    def forward(self, tree_graphs, tree_vec):
        batch_size = tree_graphs.batch_size
        device = tree_graphs.device
        tree_graphs = tree_graphs.local_var()
        tree_graphs.ndata['x'] = self.embedding(tree_graphs.ndata['wid'])
        tree_graphs.apply_edges(fn.copy_u('x', 'src_x'))
        line_tree_graphs = dgl.line_graph(tree_graphs, backtracking=False, shared=True)
        line_tree_graphs.ndata['h'] = torch.zeros(
            line_tree_graphs.num_nodes(), self.hidden_size).to(device)

        # Initialize
        pred_hiddens, pred_tree_vecs, pred_targets = [], [], []
        stop_hiddens, stop_targets = [], []
        root_ids = get_root_ids(tree_graphs)

        # Predict Root
        pred_hiddens.append(torch.zeros(batch_size, self.hidden_size).to(device))
        pred_tree_vecs.append(tree_vec)
        pred_targets.append(tree_graphs.ndata['wid'][root_ids].cpu().tolist())

        for eid, p in dfs_order(tree_graphs, root_ids):
            

        for t in range(max_iter):
            cur_x = []
            cur_h_nei, cur_o_nei = [], []

            for node_x, real_y, _ in prop_list:
                # Neighbors for message passing (target not included)
                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
                pad_len = MAX_NB - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                # Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
                pad_len = MAX_NB - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                # Current clique embedding
                cur_x.append(node_x.wid)

            # Message passing
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            # Node Aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
            cur_o = cur_o_nei.sum(dim=1)

            # Gather targets
            pred_target, pred_list = [], []
            stop_target = []
            for i, m in enumerate(prop_list):
                node_x, node_y, direction = m
                x, y = node_x.idx, node_y.idx
                h[(x, y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i)
                stop_target.append(direction)

            # Hidden states for stop prediction
            cur_batch = torch.LongTensor(batch_list)
            cur_mol_vec = tree_vec.index_select(0, cur_batch)
            stop_hidden = torch.cat([cur_x, cur_o, cur_mol_vec], dim=1)
            stop_hiddens.append(stop_hidden)
            stop_targets.extend(stop_target)

            # Hidden states for clique prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = torch.LongTensor(batch_list)
                pred_mol_vecs.append(tree_vec.index_select(0, cur_batch))

                cur_pred = torch.LongTensor(pred_list)
                pred_hiddens.append(new_h.index_select(0, cur_pred))
                pred_targets.extend(pred_target)

        # Last stop at root
        cur_x, cur_o_nei = [], []
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = torch.LongTensor(cur_x)
        cur_x = self.embedding(cur_x)
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x, cur_o, tree_vec], dim=1)
        stop_hiddens.append(stop_hidden)
        stop_targets.extend([0] * batch_size)

        # Predict next clique
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_mol_vecs = torch.cat(pred_mol_vecs, dim=0)
        pred_vecs = torch.cat([pred_hiddens, pred_mol_vecs], dim=1)
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        pred_targets = torch.LongTensor(pred_targets)

        pred_loss = self.pred_loss(pred_scores, pred_targets) / batch_size
        _, preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        # Predict stop
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_vecs = nn.ReLU()(self.U(stop_hiddens))
        stop_scores = self.U_s(stop_vecs).squeeze()
        stop_targets = torch.Tensor(stop_targets)

        stop_loss = self.stop_loss(stop_scores, stop_targets) / batch_size
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()

    def decode(self, mol_vec, prob_decode):
        stack, trace = [], []
        init_hidden = torch.zeros(1, self.hidden_size)
        zero_pad = torch.zeros(1, 1, self.hidden_size)

        # Root Prediction
        root_hidden = torch.cat([init_hidden, mol_vec], dim=1)
        root_hidden = nn.ReLU()(self.W(root_hidden))
        root_score = self.W_o(root_hidden)
        _, root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.item()

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append((root, self.vocab.get_slots(root.wid)))

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x, fa_slot = stack[-1]
            cur_h_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1, -1, self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = torch.LongTensor([node_x.wid])
            cur_x = self.embedding(cur_x)

            # Predict stop
            cur_h = cur_h_nei.sum(dim=1)
            stop_hidden = torch.cat([cur_x, cur_h, mol_vec], dim=1)
            stop_hidden = nn.ReLU()(self.U(stop_hidden))
            stop_score = nn.Sigmoid()(self.U_s(stop_hidden) * 20).squeeze()

            if prob_decode:
                backtrack = (torch.bernoulli(1.0 - stop_score.data)[0] == 1)
            else:
                backtrack = (stop_score.item() < 0.5)

            if not backtrack:  # Forward: Predict next clique
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_hidden = torch.cat([new_h, mol_vec], dim=1)
                pred_hidden = nn.ReLU()(self.W(pred_hidden))
                pred_score = nn.Softmax()(self.W_o(pred_hidden) * 20)
                if prob_decode:
                    sort_wid = torch.multinomial(pred_score.data.squeeze(), 5)
                else:
                    _, sort_wid = torch.sort(pred_score, dim=1, descending=True)
                    sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True  # No more children can be added
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = next_wid
                    node_y.idx = step + 1
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx, node_y.idx)] = new_h[0]
                    stack.append((node_y, next_slots))
                    all_nodes.append(node_y)

            if backtrack:  # Backtrack, use if instead of else
                if len(stack) == 1:
                    break  # At root, terminate

                node_fa, _ = stack[-2]
                cur_h_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1, -1, self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx, node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I',
             'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
             bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
             bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def mol2graph(mol_batch):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms, fbonds = [], [padding]  # Ensure bond is 1-indexed
    in_bonds, all_bonds = [], [(-1, -1)]  # Ensure bond is 1-indexed
    scope = []
    total_atoms = 0

    for smiles in mol_batch:
        mol = get_mol(smiles)
        # mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append(atom_features(atom))
            in_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() + total_atoms
            y = a2.GetIdx() + total_atoms

            b = len(all_bonds)
            all_bonds.append((x, y))
            fbonds.append(torch.cat([fatoms[x], bond_features(bond)], 0))
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y, x))
            fbonds.append(torch.cat([fatoms[y], bond_features(bond)], 0))
            in_bonds[x].append(b)

        scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    agraph = torch.zeros(total_atoms, MAX_NB).long()
    bgraph = torch.zeros(total_bonds, MAX_NB).long()

    for a in range(total_atoms):
        for i, b in enumerate(in_bonds[a]):
            agraph[a, i] = b

    for b1 in range(1, total_bonds):
        x, y = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1, i] = b2

    return fatoms, fbonds, agraph, bgraph, scope

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

class MPN(nn.Module):
    def __init__(self, node_feats, edge_feats, hidden_size, depth):
        super(MPN, self).__init__()

        self.W_i = nn.Linear(edge_feats, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Sequential(
            nn.Linear(node_feats + hidden_size, hidden_size),
            nn.ReLU()
        )
        self.depth = depth

    def forward(self, mol_graph):
        mol_graph = mol_graph.local_var()
        line_mol_graph = dgl.line_graph(mol_graph, backtracking=False, shared=True)

        line_input = self.W_i(line_mol_graph.ndata['x'])
        line_mol_graph.ndata['msg_input'] = line_input
        line_mol_graph.ndata['msg'] = F.relu(line_input)

        # Message passing over the line graph
        for _ in range(self.depth - 1):
            line_mol_graph.update_all(message_func=fn.copy_u('msg', 'msg'),
                                      reduce_func=fn.sum('msg', 'nei_msg'))
            nei_msg = self.W_h(line_mol_graph.ndata['nei_msg'])
            line_mol_graph.ndata['msg'] = F.relu(line_input + nei_msg)

        # Message passing over the raw graph
        mol_graph.edata['msg'] = line_mol_graph.ndata['msg']
        mol_graph.update_all(message_func=fn.copy_e('msg', 'msg'),
                             reduce_func=fn.sum('msg', 'nei_msg'))

        raw_input = torch.cat([mol_graph.ndata['x'], mol_graph.ndata['nei_msg']], dim=1)
        mol_graph.ndata['atom_hiddens'] = self.W_o(raw_input)

        # Readout
        mol_vecs = dgl.mean_nodes(mol_graph, 'atom_hiddens')

        return mol_vecs

class JTMPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, cand_batch, tree_mess):
        fatoms, fbonds = [], []
        in_bonds, all_bonds = [], []
        mess_dict, all_mess = {}, [torch.zeros(self.hidden_size)]  # Ensure index 0 is vec(0)
        total_atoms = 0
        scope = []

        for e, vec in tree_mess.iteritems():
            mess_dict[e] = len(all_mess)
            all_mess.append(vec)

        for mol, all_nodes, ctr_node in cand_batch:
            n_atoms = mol.GetNumAtoms()

            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms
                # Here x_nid,y_nid could be 0
                x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bfeature = bond_features(bond)

                b = len(all_mess) + len(all_bonds)  # bond idx offseted by len(all_mess)
                all_bonds.append((x, y))
                fbonds.append(torch.cat([fatoms[x], bfeature], 0))
                in_bonds[y].append(b)

                b = len(all_mess) + len(all_bonds)
                all_bonds.append((y, x))
                fbonds.append(torch.cat([fatoms[y], bfeature], 0))
                in_bonds[x].append(b)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid, y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid, y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid, x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid, x_bid)]
                        in_bonds[x].append(mess_idx)

            scope.append((total_atoms, n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)
        total_mess = len(all_mess)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        agraph = torch.zeros(total_atoms, MAX_NB).long()
        bgraph = torch.zeros(total_bonds, MAX_NB).long()
        tree_message = torch.stack(all_mess, dim=0)

        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a]):
                agraph[a, i] = b

        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):  # b2 is offseted by len(all_mess)
                if b2 < total_mess or all_bonds[b2 - total_mess][0] != y:
                    bgraph[b1, i] = b2

        binput = self.W_i(fbonds)
        graph_message = nn.ReLU()(binput)

        for _ in range(self.depth - 1):
            message = torch.cat([tree_message, graph_message], dim=0)
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            graph_message = nn.ReLU()(binput + nei_message)

        message = torch.cat([tree_message, graph_message], dim=0)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = nn.ReLU()(self.W_o(ainput))

        mol_vecs = []
        for st, le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

class JTNNVAE(nn.Module):
    def __init__(self, vocab, hidden_size, latent_size, depth, stereo=True):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth

        self.embedding = nn.Embedding(self.vocab.size(), hidden_size)
        self.jtnn = JTNNEncoder(self.vocab, hidden_size, self.embedding)
        self.jtmpn = JTMPN(hidden_size, depth)
        self.mpn = MPN(ATOM_FDIM, ATOM_FDIM + BOND_FDIM, hidden_size, depth)
        self.decoder = JTNNDecoder(self.vocab, hidden_size, latent_size / 2, self.embedding)

        self.T_mean = nn.Linear(hidden_size, latent_size / 2)
        self.T_var = nn.Linear(hidden_size, latent_size / 2)
        self.G_mean = nn.Linear(hidden_size, latent_size / 2)
        self.G_var = nn.Linear(hidden_size, latent_size / 2)

        self.assm_loss = nn.CrossEntropyLoss(size_average=False)
        self.use_stereo = stereo
        if stereo:
            self.stereo_loss = nn.CrossEntropyLoss(size_average=False)

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal(param)

    def encode(self, batch_tree_graphs, batch_mol_graphs):
        tree_mess, tree_vec = self.jtnn(batch_tree_graphs)
        mol_vec = self.mpn(batch_mol_graphs)
        return tree_mess, tree_vec, mol_vec

    def encode_latent_mean(self, smiles_list):
        mol_batch = [MolTree(s) for s in smiles_list]
        for mol_tree in mol_batch:
            mol_tree.recover()

        _, tree_vec, mol_vec = self.encode(mol_batch)
        tree_mean = self.T_mean(tree_vec)
        mol_mean = self.G_mean(mol_vec)
        return torch.cat([tree_mean, mol_mean], dim=1)

    def forward(self, batch_trees, batch_tree_graphs, batch_mol_graphs, beta=0):
        batch_size = batch_tree_graphs.batch_size
        tree_mess, tree_vec, mol_vec = self.encode(batch_tree_graphs, batch_mol_graphs)

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))    # Following Mueller et al.

        z_mean = torch.cat([tree_mean, mol_mean], dim=1)
        z_log_var = torch.cat([tree_log_var, mol_log_var], dim=1)
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / \
                  batch_size

        epsilon = torch.randn(batch_size, self.latent_size / 2)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = torch.randn(batch_size, self.latent_size / 2)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(batch_tree_graphs, tree_vec)
        assm_loss, assm_acc = self.assm(mol_batch, mol_vec, tree_mess)
        if self.use_stereo:
            stereo_loss, stereo_acc = self.stereo(mol_batch, mol_vec)
        else:
            stereo_loss, stereo_acc = 0, 0

        loss = word_loss + topo_loss + assm_loss + 2 * stereo_loss + beta * kl_loss

        return loss, kl_loss.item(), word_acc, topo_acc, assm_acc, stereo_acc

    def assm(self, mol_batch, mol_vec, tree_mess):
        cands = []
        batch_idx = []
        for i, mol_tree in enumerate(mol_batch):
            for node in mol_tree.nodes:
                # Leaf node's attachment is determined by neighboring node's attachment
                if node.is_leaf or len(node.cands) == 1: continue
                cands.extend([(cand, mol_tree.nodes, node) for cand in node.cand_mols])
                batch_idx.extend([i] * len(node.cands))

        cand_vec = self.jtmpn(cands, tree_mess)
        cand_vec = self.G_mean(cand_vec)

        batch_idx = torch.LongTensor(batch_idx)
        mol_vec = mol_vec.index_select(0, batch_idx)

        mol_vec = mol_vec.view(-1, 1, self.latent_size / 2)
        cand_vec = cand_vec.view(-1, self.latent_size / 2, 1)
        scores = torch.bmm(mol_vec, cand_vec).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score[label].item() >= cur_score.max().item():
                    acc += 1

                label = torch.LongTensor([label])
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        # all_loss = torch.stack(all_loss).sum() / len(mol_batch)
        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def stereo(self, mol_batch, mol_vec):
        stereo_cands, batch_idx = [], []
        labels = []
        for i, mol_tree in enumerate(mol_batch):
            cands = mol_tree.stereo_cands
            if len(cands) == 1: continue
            if mol_tree.smiles3D not in cands:
                cands.append(mol_tree.smiles3D)
            stereo_cands.extend(cands)
            batch_idx.extend([i] * len(cands))
            labels.append((cands.index(mol_tree.smiles3D), len(cands)))

        if len(labels) == 0:
            return torch.zeros(1), 1.0

        batch_idx = torch.LongTensor(batch_idx)
        stereo_cands = self.mpn(mol2graph(stereo_cands))
        stereo_cands = self.G_mean(stereo_cands)
        stereo_labels = mol_vec.index_select(0, batch_idx)
        scores = torch.nn.CosineSimilarity()(stereo_cands, stereo_labels)

        st, acc = 0, 0
        all_loss = []
        for label, le in labels:
            cur_scores = scores.narrow(0, st, le)
            if cur_scores.data[label] >= cur_scores.max().data[0]:
                acc += 1
            label = torch.LongTensor([label])
            all_loss.append(self.stereo_loss(cur_scores.view(1, -1), label))
            st += le
        # all_loss = torch.cat(all_loss).sum() / len(labels)
        all_loss = sum(all_loss) / len(labels)
        return all_loss, acc * 1.0 / len(labels)

    def reconstruct(self, smiles, prob_decode=False):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _, tree_vec, mol_vec = self.encode([mol_tree])

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))  # Following Mueller et al.

        epsilon = torch.randn(1, self.latent_size / 2)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = torch.randn(1, self.latent_size / 2)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
        return self.decode(tree_vec, mol_vec, prob_decode)

    def recon_eval(self, smiles):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _, tree_vec, mol_vec = self.encode([mol_tree])

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))  # Following Mueller et al.

        all_smiles = []
        for _ in range(10):
            epsilon = torch.randn(1, self.latent_size / 2)
            tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
            epsilon = torch.randn(1, self.latent_size / 2)
            mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
            for _ in range(10):
                new_smiles = self.decode(tree_vec, mol_vec, prob_decode=True)
                all_smiles.append(new_smiles)
        return all_smiles

    def sample_prior(self, prob_decode=False):
        tree_vec = torch.randn(1, self.latent_size / 2)
        mol_vec = torch.randn(1, self.latent_size / 2)
        return self.decode(tree_vec, mol_vec, prob_decode)

    def sample_eval(self):
        tree_vec = torch.randn(1, self.latent_size / 2)
        mol_vec = torch.randn(1, self.latent_size / 2)
        all_smiles = []
        for _ in range(100):
            s = self.decode(tree_vec, mol_vec, prob_decode=True)
            all_smiles.append(s)
        return all_smiles

    def decode(self, tree_vec, mol_vec, prob_decode):
        pred_root, pred_nodes = self.decoder.decode(tree_vec, prob_decode)

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        tree_mess = self.jtnn([pred_root])[0]

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble(tree_mess, mol_vec, pred_nodes, cur_mol, global_amap, [], pred_root, None,
                                    prob_decode)
        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        if cur_mol is None: return None
        if self.use_stereo == False:
            return Chem.MolToSmiles(cur_mol)

        smiles2D = Chem.MolToSmiles(cur_mol)
        stereo_cands = decode_stereo(smiles2D)
        if len(stereo_cands) == 1:
            return stereo_cands[0]
        stereo_vecs = self.mpn(mol2graph(stereo_cands))
        stereo_vecs = self.G_mean(stereo_vecs)
        scores = nn.CosineSimilarity()(stereo_vecs, mol_vec)
        _, max_id = scores.max(dim=0)
        return stereo_cands[max_id.data[0]]

    def dfs_assemble(self, tree_mess, mol_vec, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node,
                     prob_decode):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0:
            return None
        cand_smiles, cand_mols, cand_amap = zip(*cands)

        cands = [(candmol, all_nodes, cur_node) for candmol in cand_mols]

        cand_vecs = self.jtmpn(cands, tree_mess)
        cand_vecs = self.G_mean(cand_vecs)
        mol_vec = mol_vec.squeeze()
        scores = torch.mv(cand_vecs, mol_vec) * 20

        if prob_decode:
            probs = nn.Softmax()(scores.view(1, -1)).squeeze() + 1e-5  # prevent prob = 0
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)  # father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue

            result = True
            for nei_node in children:
                if nei_node.is_leaf: continue
                cur_mol = self.dfs_assemble(tree_mess, mol_vec, all_nodes, cur_mol, new_global_amap, pred_amap,
                                            nei_node, cur_node, prob_decode)
                if cur_mol is None:
                    result = False
                    break
            if result: return cur_mol

        return None
