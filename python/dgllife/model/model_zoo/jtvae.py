# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# JTVAE

from functools import partial

import copy
import dgl
import dgl.function as fn
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.traversal import bfs_edges_generator, dfs_labeled_edges_generator

from ...data.jtvae import get_atom_featurizer_enc, get_bond_featurizer_enc
from ...utils.featurizers import ConcatFeaturizer, atom_type_one_hot, atom_degree_one_hot,\
    atom_formal_charge_one_hot, atom_is_aromatic, bond_type_one_hot, bond_is_in_ring
from ...utils.jtvae.chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols, \
    decode_stereo, get_mol
from ...utils.mol_to_graph import mol_to_bigraph

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
        new_h = (torch.tensor(1.0).to(z.device) - z) * node.data['sum_h'] + z * pre_h
        return {'h': new_h}

# pylint: disable=R1710
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
        if 'x' not in tree_graphs.ndata:
            tree_graphs.ndata['x'] = self.embedding(tree_graphs.ndata['wid'])
        tree_graphs.apply_edges(fn.copy_u('x', 'src_x'))
        tree_graphs = tree_graphs.local_var()

        line_tree_graphs = dgl.line_graph(tree_graphs, backtracking=False)
        line_tree_graphs.ndata.update({
            'src_x': tree_graphs.edata['src_x'],
            'src_x_r': self.W_r(tree_graphs.edata['src_x']),
            # Exploit the fact that the reduce function is a sum of incoming messages,
            # and uncomputed messages are zero vectors.
            'h': torch.zeros(line_tree_graphs.num_nodes(), self.hidden_size).to(device),
            'sum_h': torch.zeros(line_tree_graphs.num_nodes(), self.hidden_size).to(device),
            'sum_gated_h': torch.zeros(line_tree_graphs.num_nodes(), self.hidden_size).to(device)
        })

        # Get the ID of the root nodes, the first node of all trees
        root_ids = get_root_ids(tree_graphs)

        for eid in level_order(tree_graphs, root_ids.to(dtype=tree_graphs.idtype)):
            line_tree_graphs.pull(v=eid, message_func=fn.copy_u('h', 'h_nei'),
                                  reduce_func=fn.sum('h_nei', 'sum_h'))
            line_tree_graphs.pull(v=eid, message_func=self.gru_message,
                                  reduce_func=fn.sum('m', 'sum_gated_h'))
            line_tree_graphs.apply_nodes(self.gru_update, v=eid)

        # Readout
        tree_graphs.ndata['h'] = torch.zeros(tree_graphs.num_nodes(), self.hidden_size).to(device)
        tree_graphs.edata['h'] = line_tree_graphs.ndata['h']
        root_ids = root_ids.to(device)
        tree_graphs.pull(v=root_ids.to(dtype=tree_graphs.idtype),
                         message_func=fn.copy_e('h', 'm'),
                         reduce_func=fn.sum('m', 'h'))
        root_vec = torch.cat([
            tree_graphs.ndata['x'][root_ids],
            tree_graphs.ndata['h'][root_ids]
        ], dim=1)
        root_vec = self.W(root_vec)

        return tree_graphs.edata['h'], root_vec

def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i, s1 in enumerate(fa_slots):
        a1, c1, h1 = s1
        for j, s2 in enumerate(ch_slots):
            a2, c2, h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append((i, j))

    if len(matches) == 0:
        return False

    fa_match, ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True

def can_assemble(node_x, node_y):
    neis = node_x['neighbors'] + [node_y]
    for i, nei in enumerate(neis):
        nei['nid'] = i

    neighbors = [nei for nei in neis if nei['mol'].GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x: x['mol'].GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei['mol'].GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands = enum_assemble(node_x, neighbors)
    return len(cands) > 0

def dfs_order(forest, roots):
    edges = dfs_labeled_edges_generator(forest, roots, has_reverse_edge=True)
    for e, l in zip(*edges):
        # Exploit the fact that the reverse edge ID equals to 1 xor forward
        # edge ID. Normally, this should be done using find_edges().
        yield e ^ l, l

def mol_tree_node(smiles, wid=None, idx=None, nbrs=None):
    if nbrs is None:
        nbrs = []
    return {'smiles': smiles, 'mol': get_mol(smiles), 'wid': wid, 'idx': idx, 'neighbors': nbrs}

def gru_functional(x, h_nei, wz, wr, ur, wh):
    hidden_size = x.size()[-1]
    sum_h = h_nei.sum(dim=1)
    z_input = torch.cat([x, sum_h], dim=1)
    z = torch.sigmoid(wz(z_input))

    r_1 = wr(x).view(-1, 1, hidden_size)
    r_2 = ur(h_nei)
    r = torch.sigmoid(r_1 + r_2)

    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x, sum_gated_h], dim=1)
    pre_h = torch.tanh(wh(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h

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
        self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, tree_graphs, tree_vec):
        device = tree_vec.device
        batch_size = tree_graphs.batch_size

        root_ids = get_root_ids(tree_graphs)

        if 'x' not in tree_graphs.ndata:
            tree_graphs.ndata['x'] = self.embedding(tree_graphs.ndata['wid'])
        if 'src_x' not in tree_graphs.edata:
            tree_graphs.apply_edges(fn.copy_u('x', 'src_x'))
        tree_graphs = tree_graphs.local_var()
        tree_graphs.apply_edges(func=lambda edges: {'dst_wid': edges.dst['wid']})

        line_tree_graphs = dgl.line_graph(tree_graphs, backtracking=False, shared=True)
        line_num_nodes = line_tree_graphs.num_nodes()
        line_tree_graphs.ndata.update({
            'src_x_r': self.W_r(line_tree_graphs.ndata['src_x']),
            # Exploit the fact that the reduce function is a sum of incoming messages,
            # and uncomputed messages are zero vectors.
            'h': torch.zeros(line_num_nodes, self.hidden_size).to(device),
            'vec': dgl.broadcast_edges(tree_graphs, tree_vec),
            'sum_h': torch.zeros(line_num_nodes, self.hidden_size).to(device),
            'sum_gated_h': torch.zeros(line_num_nodes, self.hidden_size).to(device)
        })

        # input tensors for stop prediction (p) and label prediction (q)
        pred_hiddens, pred_mol_vecs, pred_targets = [], [], []
        stop_hiddens, stop_targets = [], []

        # Predict root
        pred_hiddens.append(torch.zeros(batch_size, self.hidden_size).to(device))
        pred_targets.append(tree_graphs.ndata['wid'][root_ids.to(device)])
        pred_mol_vecs.append(tree_vec)

        # Traverse the tree and predict on children
        for eid, p in dfs_order(tree_graphs, root_ids.to(dtype=tree_graphs.idtype)):
            eid = eid.to(device)
            p = p.to(device=device, dtype=tree_graphs.idtype)

            # Message passing excluding the target
            line_tree_graphs.pull(v=eid, message_func=fn.copy_u('h', 'h_nei'),
                                  reduce_func=fn.sum('h_nei', 'sum_h'))
            line_tree_graphs.pull(v=eid, message_func=self.gru_message,
                                  reduce_func=fn.sum('m', 'sum_gated_h'))
            line_tree_graphs.apply_nodes(self.gru_update, v=eid)

            # Node aggregation including the target
            # By construction, the edges of the raw graph follow the order of
            # (i1, j1), (j1, i1), (i2, j2), (j2, i2), ... The order of the nodes
            # in the line graph corresponds to the order of the edges in the raw graph.
            eid = eid.long()
            reverse_eid = torch.bitwise_xor(eid, torch.tensor(1).to(device))
            cur_o = line_tree_graphs.ndata['sum_h'][eid] + \
                    line_tree_graphs.ndata['h'][reverse_eid]

            # Gather targets
            mask = (p == torch.tensor(0).to(device))
            pred_list = eid[mask]
            stop_target = torch.tensor(1).to(device) - p

            # Hidden states for stop prediction
            stop_hidden = torch.cat([line_tree_graphs.ndata['src_x'][eid],
                                     cur_o, line_tree_graphs.ndata['vec'][eid]], dim=1)
            stop_hiddens.append(stop_hidden)
            stop_targets.extend(stop_target)

            #Hidden states for clique prediction
            if len(pred_list) > 0:
                pred_mol_vecs.append(line_tree_graphs.ndata['vec'][pred_list])
                pred_hiddens.append(line_tree_graphs.ndata['h'][pred_list])
                pred_targets.append(line_tree_graphs.ndata['dst_wid'][pred_list])

        #Last stop at root
        root_ids = root_ids.to(device)
        cur_x = tree_graphs.ndata['x'][root_ids]
        tree_graphs.edata['h'] = line_tree_graphs.ndata['h']
        tree_graphs.pull(v=root_ids.to(dtype=tree_graphs.idtype),
                         message_func=fn.copy_e('h', 'm'), reduce_func=fn.sum('m', 'cur_o'))
        stop_hidden = torch.cat([cur_x, tree_graphs.ndata['cur_o'][root_ids], tree_vec], dim=1)
        stop_hiddens.append(stop_hidden)
        stop_targets.extend(torch.zeros(batch_size).to(device))

        # Predict next clique
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_mol_vecs = torch.cat(pred_mol_vecs, dim=0)
        pred_vecs = torch.cat([pred_hiddens, pred_mol_vecs], dim=1)
        pred_vecs = F.relu(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        pred_targets = torch.cat(pred_targets, dim=0)

        pred_loss = self.pred_loss(pred_scores, pred_targets) / batch_size
        _, preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        # Predict stop
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_vecs = F.relu(self.U(stop_hiddens))
        stop_scores = self.U_s(stop_vecs).squeeze()
        stop_targets = torch.Tensor(stop_targets).to(device)

        stop_loss = self.stop_loss(stop_scores, stop_targets) / batch_size
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()

    def decode(self, mol_vec, prob_decode):
        device = mol_vec.device
        stack = []
        init_hidden = torch.zeros(1, self.hidden_size).to(device)
        zero_pad = torch.zeros(1, 1, self.hidden_size).to(device)

        # Root Prediction
        root_hidden = torch.cat([init_hidden, mol_vec], dim=1)
        root_hidden = F.relu(self.W(root_hidden))
        root_score = self.W_o(root_hidden)
        _, root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.item()

        root = mol_tree_node(smiles=self.vocab.get_smiles(root_wid), wid=root_wid, idx=0)
        stack.append((root, self.vocab.get_slots(root['wid'])))

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x, fa_slot = stack[-1]
            cur_h_nei = [h[(node_y['idx'], node_x['idx'])] for node_y in node_x['neighbors']]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1, -1, self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = torch.LongTensor([node_x['wid']]).to(device)
            cur_x = self.embedding(cur_x)

            # Predict stop
            cur_h = cur_h_nei.sum(dim=1)
            stop_hidden = torch.cat([cur_x, cur_h, mol_vec], dim=1)
            stop_hidden = F.relu(self.U(stop_hidden))
            stop_score = torch.sigmoid(self.U_s(stop_hidden) * 20).squeeze()

            if prob_decode:
                backtrack = (torch.bernoulli(1.0 - stop_score.data)[0] == 1)
            else:
                backtrack = (stop_score.item() < 0.5)

            if not backtrack:  # Forward: Predict next clique
                new_h = gru_functional(cur_x, cur_h_nei, self.gru_update.W_z, self.W_r,
                                       self.gru_message.U_r, self.gru_update.W_h)
                pred_hidden = torch.cat([new_h, mol_vec], dim=1)
                pred_hidden = F.relu(self.W(pred_hidden))
                pred_score = torch.softmax(self.W_o(pred_hidden) * 20, dim=1)
                if prob_decode:
                    sort_wid = torch.multinomial(pred_score.data.squeeze(), 5)
                else:
                    _, sort_wid = torch.sort(pred_score, dim=1, descending=True)
                    sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = mol_tree_node(smiles=self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True  # No more children can be added
                else:
                    node_y = mol_tree_node(smiles=self.vocab.get_smiles(next_wid),
                                           wid=next_wid, idx=step + 1, nbrs=[node_x])
                    h[(node_x['idx'], node_y['idx'])] = new_h[0]
                    stack.append((node_y, next_slots))
                    all_nodes.append(node_y)

            if backtrack:  # Backtrack, use if instead of else
                if len(stack) == 1:
                    break  # At root, terminate

                node_fa, _ = stack[-2]
                cur_h_nei = [h[(node_y['idx'], node_x['idx'])] for node_y in node_x['neighbors']
                             if node_y['idx'] != node_fa['idx']]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1, -1, self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                new_h = gru_functional(cur_x, cur_h_nei, self.gru_update.W_z, self.W_r,
                                       self.gru_message.U_r, self.gru_update.W_h)
                h[(node_x['idx'], node_fa['idx'])] = new_h[0]
                node_fa['neighbors'].append(node_x)
                stack.pop()

        return root, all_nodes

class MPN(nn.Module):
    def __init__(self, hidden_size, depth, in_node_feats=39, in_edge_feats=50):
        super(MPN, self).__init__()

        self.W_i = nn.Linear(in_edge_feats, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Sequential(
            nn.Linear(in_node_feats + hidden_size, hidden_size),
            nn.ReLU()
        )
        self.depth = depth

    def forward(self, mol_graph):
        mol_graph = mol_graph.local_var()
        line_mol_graph = dgl.line_graph(mol_graph, backtracking=False)

        line_input = self.W_i(mol_graph.edata['x'])
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

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

class JTMPN(nn.Module):

    def __init__(self, hidden_size, depth, in_node_feats=35, in_edge_feats=40):
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(in_edge_feats, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(in_node_feats + hidden_size, hidden_size)
        self.atom_featurizer = ConcatFeaturizer([
            partial(atom_type_one_hot,
                    allowable_set=['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                   'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn'],
                    encode_unknown=True),
            partial(atom_degree_one_hot, allowable_set=[0, 1, 2, 3, 4], encode_unknown=True),
            partial(atom_formal_charge_one_hot, allowable_set=[-1, -2, 1, 2],
                    encode_unknown=True),
            atom_is_aromatic
        ])
        self.bond_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])

    def forward(self, cand_batch, tree_mess, device='cpu'):
        fatoms, fbonds = [], []
        in_bonds, all_bonds = [], []
        # Ensure index 0 is vec(0)
        mess_dict, all_mess = {}, [torch.zeros(self.hidden_size).to(device)]
        total_atoms = 0
        scope = []

        for e, vec in tree_mess.items():
            mess_dict[e] = len(all_mess)
            all_mess.append(vec)

        for mol, all_nodes, _ in cand_batch:
            n_atoms = mol.GetNumAtoms()

            for atom in mol.GetAtoms():
                fatoms.append(torch.Tensor(self.atom_featurizer(atom)))
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms
                # Here x_nid,y_nid could be 0
                x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1]['idx'] if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1]['idx'] if y_nid > 0 else -1

                bfeature = torch.Tensor(self.bond_featurizer(bond))

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
        fatoms = torch.stack(fatoms, 0).to(device)
        fbonds = torch.stack(fbonds, 0).to(device)
        agraph = torch.zeros(total_atoms, MAX_NB).long().to(device)
        bgraph = torch.zeros(total_bonds, MAX_NB).long().to(device)
        tree_message = torch.stack(all_mess, dim=0)

        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a]):
                if i == MAX_NB:
                    break
                agraph[a, i] = b

        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):  # b2 is offseted by len(all_mess)
                if i == MAX_NB:
                    break
                if b2 < total_mess or all_bonds[b2 - total_mess][0] != y:
                    bgraph[b1, i] = b2

        binput = self.W_i(fbonds)
        graph_message = F.relu(binput)

        for i in range(self.depth - 1):
            message = torch.cat([tree_message, graph_message], dim=0)
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            graph_message = F.relu(binput + nei_message)

        message = torch.cat([tree_message, graph_message], dim=0)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))

        mol_vecs = []
        for st, le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

class JTNNVAE(nn.Module):
    # TODO
    def __init__(self, vocab, hidden_size, latent_size, depth, stereo=True):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth

        self.embedding = nn.Embedding(self.vocab.size(), hidden_size)
        self.jtnn = JTNNEncoder(self.vocab, hidden_size, self.embedding)
        self.jtmpn = JTMPN(hidden_size, depth)
        self.mpn = MPN(hidden_size, depth)
        self.decoder = JTNNDecoder(self.vocab, hidden_size, latent_size // 2, self.embedding)

        self.T_mean = nn.Linear(hidden_size, latent_size // 2)
        self.T_var = nn.Linear(hidden_size, latent_size // 2)
        self.G_mean = nn.Linear(hidden_size, latent_size // 2)
        self.G_var = nn.Linear(hidden_size, latent_size // 2)

        self.assm_loss = nn.CrossEntropyLoss(reduction='sum')
        self.use_stereo = stereo
        if stereo:
            self.stereo_loss = nn.CrossEntropyLoss(reduction='sum')

        self.atom_featurizer = get_atom_featurizer_enc()
        self.bond_featurizer = get_bond_featurizer_enc()

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param.data, 0)
            else:
                nn.init.xavier_normal_(param.data)

    def encode(self, batch_tree_graphs, batch_mol_graphs):
        tree_mess, tree_vec = self.jtnn(batch_tree_graphs)
        mol_vec = self.mpn(batch_mol_graphs)
        return tree_mess, tree_vec, mol_vec

    def forward(self, batch_trees, batch_tree_graphs, batch_mol_graphs, stereo_cand_batch_idx,
                stereo_cand_labels, batch_stereo_cand_graphs, beta=0):
        batch_size = batch_tree_graphs.batch_size
        device = batch_tree_graphs.device
        tree_mess, tree_vec, mol_vec = self.encode(batch_tree_graphs, batch_mol_graphs)

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))    # Following Mueller et al.

        z_mean = torch.cat([tree_mean, mol_mean], dim=1)
        z_log_var = torch.cat([tree_log_var, mol_log_var], dim=1)
        kl_loss = -0.5 * torch.sum(torch.tensor(1.0).to(device) + z_log_var - z_mean * z_mean -
                                   torch.exp(z_log_var)) / batch_size

        epsilon = torch.randn(batch_size, self.latent_size // 2).to(device)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon

        epsilon = torch.randn(batch_size, self.latent_size // 2).to(device)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(batch_tree_graphs, tree_vec)
        assm_loss, assm_acc = self.assm(batch_trees, batch_tree_graphs, mol_vec, tree_mess)

        if self.use_stereo:
            stereo_loss, stereo_acc = self.stereo(stereo_cand_batch_idx, stereo_cand_labels,
                                                  batch_stereo_cand_graphs, mol_vec)
        else:
            stereo_loss, stereo_acc = torch.tensor(0.).to(device), 0

        loss = word_loss + topo_loss + assm_loss + 2 * stereo_loss + beta * kl_loss

        return loss, kl_loss.item(), word_acc, topo_acc, assm_acc, stereo_acc

    def edata_to_dict(self, g, tree_mess):
        tree_mess_ = dict()
        src, dst = g.edges()
        for i, edge in enumerate(tuple(zip(src.tolist(), dst.tolist()))):
            tree_mess_[edge] = tree_mess[i]

        return tree_mess_

    def assm(self, batch_trees, tree_graphs, mol_vec, tree_mess):
        device = tree_graphs.device

        cands = []
        cand_batch_idx = []
        for i, tree in enumerate(batch_trees):
            for _, node in tree.nodes_dict.items():
                # Leaf node's attachment is determined by neighboring node's attachment
                if node['is_leaf'] or len(node['cands']) == 1:
                    continue
                cands.extend([(cand, tree.nodes_dict, node) for cand in node['cand_mols']])
                cand_batch_idx.extend([i] * len(node['cands']))

        tree_mess = self.edata_to_dict(tree_graphs, tree_mess)
        cand_vec = self.jtmpn(cands, tree_mess, device)
        cand_vec = self.G_mean(cand_vec)

        if len(cand_batch_idx) == 0:
            cand_batch_idx = torch.zeros(0).long().to(device)
        else:
            cand_batch_idx = torch.LongTensor(cand_batch_idx).to(device)

        mol_vec = mol_vec[cand_batch_idx]

        mol_vec = mol_vec.view(-1, 1, self.latent_size // 2)
        cand_vec = cand_vec.view(-1, self.latent_size // 2, 1)
        scores = torch.bmm(mol_vec, cand_vec).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for tree in batch_trees:
            for _, node in tree.nodes_dict.items():
                num_cands = len(node['cands'])
                if node['is_leaf'] or num_cands == 1:
                    continue

                cnt += 1
                label = node['cands'].index(node['label'])
                cur_score = scores.narrow(0, tot, num_cands)
                tot += num_cands

                if cur_score[label].item() >= cur_score.max().item():
                    acc += 1

                label = torch.LongTensor([label]).to(device)
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        if len(all_loss) > 0:
            all_loss = sum(all_loss) / len(batch_trees)
        else:
            all_loss = torch.zeros(1).to(device)
        return all_loss, acc * 1.0 / cnt

    def stereo(self, batch_idx, batch_labels, batch_stereo_cand_graphs, mol_vec):
        device = batch_stereo_cand_graphs.device

        if len(batch_labels) == 0:
            return torch.zeros(1).to(device), 1.0

        stereo_cands = self.mpn(batch_stereo_cand_graphs)
        stereo_cands = self.G_mean(stereo_cands)
        stereo_labels = mol_vec[batch_idx]
        scores = nn.CosineSimilarity()(stereo_cands, stereo_labels)

        st, acc = 0, 0
        all_loss = []
        for label, le in batch_labels:
            cur_scores = scores.narrow(0, st, le)
            if cur_scores[label].item() >= cur_scores.max().item():
                acc += 1
            label = torch.LongTensor([label]).to(device)
            all_loss.append(self.stereo_loss(cur_scores.view(1, -1), label))
            st += le

        all_loss = sum(all_loss) / len(batch_labels)
        return all_loss, acc * 1.0 / len(batch_labels)

    def reconstruct(self, tree_graph, mol_graph, prob_decode=False):
        device = tree_graph.device
        _, tree_vec, mol_vec = self.encode(tree_graph, mol_graph)

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))  # Following Mueller et al.

        epsilon = torch.randn(1, self.latent_size // 2).to(device)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = torch.randn(1, self.latent_size // 2).to(device)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
        return self.decode(tree_vec, mol_vec, prob_decode)

    def sample_prior(self, prob_decode=False):
        tree_vec = torch.randn(1, self.latent_size // 2)
        mol_vec = torch.randn(1, self.latent_size // 2)
        return self.decode(tree_vec, mol_vec, prob_decode)

    def sample_eval(self):
        tree_vec = torch.randn(1, self.latent_size // 2)
        mol_vec = torch.randn(1, self.latent_size // 2)
        all_smiles = []
        for _ in range(100):
            s = self.decode(tree_vec, mol_vec, prob_decode=True)
            all_smiles.append(s)
        return all_smiles

    def decode(self, tree_vec, mol_vec, prob_decode):
        device = tree_vec.device
        pred_root, pred_nodes = self.decoder.decode(tree_vec, prob_decode)

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node['nid'] = i + 1
            node['is_leaf'] = (len(node['neighbors']) == 1)
            if len(node['neighbors']) > 1:
                set_atommap(node['mol'], node['nid'])

        src = []
        dst = []
        for node in pred_nodes:
            cur_id = node['idx']
            for nbr in node['neighbors']:
                nbr_id = nbr['idx']
                src.extend([cur_id])
                dst.extend([nbr_id])
        if len(src) == 0:
            tree_graph = dgl.graph((src, dst), idtype=torch.int32, device=device,
                                   num_nodes=max([node['idx'] + 1 for node in pred_nodes]))
        else:
            tree_graph = dgl.graph((src, dst), idtype=torch.int32, device=device)
        node_ids = torch.LongTensor([node['idx'] for node in pred_nodes]).to(device)
        node_wid = torch.LongTensor([node['wid'] for node in pred_nodes]).to(device)
        tree_graph_x = torch.zeros(tree_graph.num_nodes(), self.hidden_size).to(device)
        tree_graph_x[node_ids] = self.embedding(node_wid)
        tree_graph.ndata['x'] = tree_graph_x
        tree_mess = self.jtnn(tree_graph)[0]
        tree_mess = self.edata_to_dict(tree_graph, tree_mess)

        cur_mol = copy_edit_mol(pred_root['mol'])
        global_amap = [{}] + [{} for _ in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble(tree_mess, mol_vec, pred_nodes, cur_mol, global_amap, [],
                                    pred_root, None, prob_decode)
        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        if cur_mol is None:
            return None
        if not self.use_stereo:
            return Chem.MolToSmiles(cur_mol)

        smiles2D = Chem.MolToSmiles(cur_mol)
        stereo_cands = decode_stereo(smiles2D)
        if len(stereo_cands) == 1:
            return stereo_cands[0]

        stereo_cand_graphs = []
        for cand in stereo_cands:
            cand = get_mol(cand)
            cg = mol_to_bigraph(cand, node_featurizer=self.atom_featurizer,
                                edge_featurizer=self.bond_featurizer,
                                canonical_atom_order=False)
            cg.apply_edges(fn.copy_u('x', 'src'))
            cg.edata['x'] = torch.cat([cg.edata.pop('src'), cg.edata['x']], dim=1)
            stereo_cand_graphs.append(cg)
        stereo_cand_graphs = dgl.batch(stereo_cand_graphs).to(device)

        stereo_vecs = self.mpn(stereo_cand_graphs)
        stereo_vecs = self.G_mean(stereo_vecs)
        scores = nn.CosineSimilarity()(stereo_vecs, mol_vec)
        _, max_id = scores.max(dim=0)
        return stereo_cands[max_id.item()]

    def dfs_assemble(self, tree_mess, mol_vec, all_nodes, cur_mol, global_amap, fa_amap,
                     cur_node, fa_node, prob_decode):
        fa_nid = fa_node['nid'] if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node['neighbors'] if nei['nid'] != fa_nid]
        neighbors = [nei for nei in children if nei['mol'].GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x['mol'].GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei['mol'].GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node['nid']]
        cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0:
            return None
        _, cand_mols, cand_amap = zip(*cands)

        cands = [(candmol, all_nodes, cur_node) for candmol in cand_mols]

        cand_vecs = self.jtmpn(cands, tree_mess, mol_vec.device)
        cand_vecs = self.G_mean(cand_vecs)
        mol_vec = mol_vec.squeeze()
        scores = torch.mv(cand_vecs, mol_vec) * 20

        if prob_decode:
            probs = torch.softmax(scores.view(1, -1)).squeeze() + 1e-5  # prevent prob = 0
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
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node['nid']][ctr_atom]

            # father is already attached
            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            result = True
            for nei_node in children:
                if nei_node['is_leaf']:
                    continue
                cur_mol = self.dfs_assemble(tree_mess, mol_vec, all_nodes, cur_mol,
                                            new_global_amap, pred_amap, nei_node,
                                            cur_node, prob_decode)
                if cur_mol is None:
                    result = False
                    break
            if result:
                return cur_mol

        return None
