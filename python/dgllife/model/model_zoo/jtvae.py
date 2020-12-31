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
    decode_stereo
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
        new_h = (torch.tensor(1.0).to(z.device) - z) * node.data['sum_h'] + z * pre_h
        return {'h': new_h}

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
        tree_graphs = tree_graphs.local_var()
        tree_graphs.ndata['x'] = self.embedding(tree_graphs.ndata['wid'])
        tree_graphs.apply_edges(fn.copy_u('x', 'src_x'))

        line_tree_graphs = dgl.line_graph(tree_graphs, backtracking=False, shared=True)
        line_tree_graphs.ndata.update({
            'src_x_r': self.W_r(line_tree_graphs.ndata['src_x']),
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
        self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, tree_graphs, tree_vec):
        device = tree_vec.device
        batch_size = tree_graphs.batch_size
        tree_graphs = tree_graphs.local_var()

        root_ids = get_root_ids(tree_graphs)

        tree_graphs.ndata['x'] = self.embedding(tree_graphs.ndata['wid'])
        tree_graphs.apply_edges(fn.copy_u('x', 'src_x'))
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

    def __init__(self, hidden_size, depth, in_node_feats=35, in_edge_feats=40):
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(in_edge_feats, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(in_node_feats + hidden_size, hidden_size)

    def forward(self, cand_graphs, tree_graphs, tree_mess,
                tree_mess_source_edges, tree_mess_target_edges):
        cand_graphs = cand_graphs.local_var()
        binput = self.W_i(cand_graphs.edata['x'])
        cand_graphs.edata['g_m'] = F.relu(binput)

        if tree_mess_source_edges.shape[0] > 0:
            src_u, src_v = tree_mess_source_edges.unbind(1)
            tgt_u, tgt_v = tree_mess_target_edges.unbind(1)
            eid = tree_graphs.edge_ids(src_u, src_v).long()
            cand_graphs.edges[tgt_u, tgt_v].data['t_m'] = tree_mess[eid]

        line_cand_graphs = dgl.line_graph(cand_graphs, backtracking=False, shared=True)

        for i in range(self.depth - 1):
            line_cand_graphs.update_all(message_func=fn.copy_u('g_m', 'm'),
                                        reduce_func=fn.sum('m', 'g_m'))
            line_cand_graphs.update_all(message_func=fn.copy_u('t_m', 'm'),
                                        reduce_func=fn.sum('m', 'g_m_2'))
            nei_message = line_cand_graphs.ndata['g_m'] + line_cand_graphs.ndata['g_m_2']
            line_cand_graphs.ndata['g_m'] = F.relu(binput + self.W_h(nei_message))

        cand_graphs.edata['g_m'] = line_cand_graphs.ndata['g_m']
        cand_graphs.update_all(fn.copy_e('g_m', 'm'), fn.sum('m', 'nei1'))
        cand_graphs.update_all(fn.copy_e('t_m', 'm'), fn.sum('m', 'nei2'))
        nei_message = cand_graphs.ndata['nei1'] + cand_graphs.ndata['nei2']
        ainput = torch.cat([cand_graphs.ndata['x'], nei_message], dim=1)
        cand_graphs.ndata['atom_hiddens'] = F.relu(self.W_o(ainput))

        return dgl.mean_nodes(cand_graphs, 'atom_hiddens')

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

    def encode_latent_mean(self, smiles_list):
        mol_batch = [MolTree(s) for s in smiles_list]
        for mol_tree in mol_batch:
            mol_tree.recover()

        _, tree_vec, mol_vec = self.encode(mol_batch)
        tree_mean = self.T_mean(tree_vec)
        mol_mean = self.G_mean(mol_vec)
        return torch.cat([tree_mean, mol_mean], dim=1)

    def forward(self, batch_trees, batch_tree_graphs, batch_mol_graphs, cand_batch_idx,
                batch_cand_graphs, tree_mess_source_edges, tree_mess_target_edges,
                stereo_cand_batch_idx, stereo_cand_labels, batch_stereo_cand_graphs, beta=0):
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
        assm_loss, assm_acc = self.assm(batch_trees, cand_batch_idx, batch_cand_graphs,
                                        batch_tree_graphs, tree_mess_source_edges,
                                        tree_mess_target_edges, mol_vec, tree_mess)

        if self.use_stereo:
            stereo_loss, stereo_acc = self.stereo(stereo_cand_batch_idx, stereo_cand_labels,
                                                  batch_stereo_cand_graphs, mol_vec)
        else:
            stereo_loss, stereo_acc = torch.tensor(0.).to(device), 0

        loss = word_loss + topo_loss + assm_loss + 2 * stereo_loss + beta * kl_loss

        return loss, kl_loss.item(), word_acc, topo_acc, assm_acc, stereo_acc

    def assm(self, batch_trees, cand_batch_idx, cand_graphs, tree_graphs, tree_mess_source_edges,
             tree_mess_target_edges, mol_vec, tree_mess):
        device = cand_graphs.device
        cand_vec = self.jtmpn(cand_graphs, tree_graphs, tree_mess,
                              tree_mess_source_edges, tree_mess_target_edges)
        cand_vec = self.G_mean(cand_vec)

        mol_vec = mol_vec[cand_batch_idx]

        mol_vec = mol_vec.view(-1, 1, self.latent_size // 2)
        cand_vec = cand_vec.view(-1, self.latent_size // 2, 1)
        scores = torch.bmm(mol_vec, cand_vec).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, tree in enumerate(batch_trees):
            for i, node in tree.nodes_dict.items():
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

    def reconstruct(self, smiles, prob_decode=False):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _, tree_vec, mol_vec = self.encode([mol_tree])

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))  # Following Mueller et al.
        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))  # Following Mueller et al.

        epsilon = torch.randn(1, self.latent_size // 2)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = torch.randn(1, self.latent_size // 2)
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
            epsilon = torch.randn(1, self.latent_size // 2)
            tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
            epsilon = torch.randn(1, self.latent_size // 2)
            mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
            for _ in range(10):
                new_smiles = self.decode(tree_vec, mol_vec, prob_decode=True)
                all_smiles.append(new_smiles)
        return all_smiles

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
        pred_root, pred_nodes = self.decoder.decode(tree_vec, prob_decode)

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        tree_mess = self.jtnn([pred_root])[0]

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for _ in pred_nodes]
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
