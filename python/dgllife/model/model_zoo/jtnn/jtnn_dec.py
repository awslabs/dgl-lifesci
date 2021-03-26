# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=C0111, C0103, E1101, W0611, W0612, W0221, E1102

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl import batch, dfs_labeled_edges_generator

from .nnutils import GRUUpdate
from ....data.jtvae import get_mol, enum_assemble_nx, DGLMolTree

def dfs_order(forest, roots):
    edges = dfs_labeled_edges_generator(forest, roots, has_reverse_edge=True)
    for e, l in zip(*edges):
        # I exploited the fact that the reverse edge ID equal to 1 xor forward
        # edge ID for molecule trees.  Normally, I should locate reverse edges
        # using find_edges().
        yield e ^ l, l

dec_tree_node_msg = fn.copy_edge(edge='m', out='m')
dec_tree_node_reduce = fn.sum(msg='m', out='h')

def dec_tree_node_update(nodes):
    return {'new': nodes.data['new'].clone().zero_()}

dec_tree_edge_msg1 = fn.copy_src(src='m', out='m')
dec_tree_edge_msg2 = fn.copy_src(src='rm', out='rm')
dec_tree_edge_reduce1 = fn.sum(msg='m', out='s')
dec_tree_edge_reduce2 = fn.sum(msg='rm', out='accum_rm')

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

    fa_match, ch_match = list(zip(*matches))
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2:  # never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2:  # never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True

def can_assemble(mol_tree, u, v_node_dict):
    u_node_dict = mol_tree.nodes_dict[u]
    u_neighbors = mol_tree.g.successors(u)
    u_neighbors_node_dict = [
        mol_tree.nodes_dict[_u]
        for _u in u_neighbors
        if _u in mol_tree.nodes_dict
    ]
    neis = u_neighbors_node_dict + [v_node_dict]
    for i, nei in enumerate(neis):
        nei['nid'] = i

    neighbors = [nei for nei in neis if nei['mol'].GetNumAtoms() > 1]
    neighbors = sorted(
        neighbors, key=lambda x: x['mol'].GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei['mol'].GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands = enum_assemble_nx(u_node_dict, neighbors)
    return len(cands) > 0

def create_node_dict(smiles, clique=None):
    if clique is None:
        clique = []
    return dict(
        smiles=smiles,
        mol=get_mol(smiles),
        clique=clique,
    )

class DGLJTNNDecoder(nn.Module):
    def __init__(self, vocab, hidden_size, latent_size, embedding):
        nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        self.embedding = embedding
        self.dec_tree_edge_update = GRUUpdate(hidden_size)

        self.W = nn.Linear(latent_size + hidden_size, hidden_size)
        self.U = nn.Linear(latent_size + 2 * hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_s = nn.Linear(hidden_size, 1)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.embedding.reset_parameters()
        self.dec_tree_edge_update.reset_parameters()
        self.W.reset_parameters()
        self.U.reset_parameters()
        self.W_o.reset_parameters()
        self.U_s.reset_parameters()

    def forward(self, mol_trees, tree_vec):
        '''
        The training procedure which computes the prediction loss given the
        ground truth tree
        '''
        mol_tree_batch = batch(mol_trees)
        mol_tree_batch_lg = dgl.line_graph(mol_tree_batch, backtracking=False, shared=True)
        mol_tree_batch_lg._node_frames = mol_tree_batch._edge_frames
        n_trees = len(mol_trees)

        return self.run(mol_tree_batch, mol_tree_batch_lg, n_trees, tree_vec)

    def run(self, mol_tree_batch, mol_tree_batch_lg, n_trees, tree_vec):
        device = tree_vec.device

        node_offset = np.cumsum([0] + mol_tree_batch.batch_num_nodes)
        root_ids = node_offset[:-1]
        n_nodes = mol_tree_batch.num_nodes()
        n_edges = mol_tree_batch.num_edges()

        mol_tree_batch.ndata.update({
            'x': self.embedding(mol_tree_batch.ndata['wid']),
            'h': torch.zeros(n_nodes, self.hidden_size).to(device),
            # whether it's newly generated node
            'new': torch.ones(n_nodes).bool().to(device),
        })

        mol_tree_batch.edata.update({
            's': torch.zeros(n_edges, self.hidden_size).to(device),
            'm': torch.zeros(n_edges, self.hidden_size).to(device),
            'r': torch.zeros(n_edges, self.hidden_size).to(device),
            'z': torch.zeros(n_edges, self.hidden_size).to(device),
            'src_x': torch.zeros(n_edges, self.hidden_size).to(device),
            'dst_x': torch.zeros(n_edges, self.hidden_size).to(device),
            'rm': torch.zeros(n_edges, self.hidden_size).to(device),
            'accum_rm': torch.zeros(n_edges, self.hidden_size).to(device),
        })

        mol_tree_batch.apply_edges(
            func=lambda edges: {
                'src_x': edges.src['x'], 'dst_x': edges.dst['x']},
        )

        # input tensors for stop prediction (p) and label prediction (q)
        p_inputs = []
        p_targets = []
        q_inputs = []
        q_targets = []

        # Predict root
        mol_tree_batch.pull(
            root_ids,
            dec_tree_node_msg,
            dec_tree_node_reduce,
            dec_tree_node_update,
        )
        # Extract hidden states and store them for stop/label prediction
        h = mol_tree_batch.nodes[root_ids].data['h']
        x = mol_tree_batch.nodes[root_ids].data['x']
        p_inputs.append(torch.cat([x, h, tree_vec], 1))
        # If the out degree is 0 we don't generate any edges at all
        root_out_degrees = mol_tree_batch.out_degrees(root_ids)
        q_inputs.append(torch.cat([h, tree_vec], 1))
        q_targets.append(mol_tree_batch.nodes[root_ids].data['wid'])

        # Traverse the tree and predict on children
        for eid, p in dfs_order(mol_tree_batch, root_ids):
            u, v = mol_tree_batch.find_edges(eid)

            p_target_list = torch.zeros_like(root_out_degrees)
            p_target_list[root_out_degrees > 0] = 1 - p
            p_target_list = p_target_list[root_out_degrees >= 0]
            p_targets.append(p_target_list.clone().detach())

            root_out_degrees -= (root_out_degrees == 0).long()
            root_out_degrees -= torch.tensor(np.isin(root_ids,
                                                     v).astype('int64'))

            mol_tree_batch_lg.pull(
                eid,
                dec_tree_edge_msg1,
                dec_tree_edge_reduce1
            )
            mol_tree_batch_lg.pull(
                eid,
                dec_tree_edge_msg2,
                dec_tree_edge_reduce2,
                self.dec_tree_edge_update
            )

            is_new = mol_tree_batch.nodes[v].data['new']
            mol_tree_batch.pull(
                v,
                dec_tree_node_msg,
                dec_tree_node_reduce,
                dec_tree_node_update,
            )
            # Extract
            n_repr = mol_tree_batch.nodes[v].data
            h = n_repr['h']
            x = n_repr['x']
            tree_vec_set = tree_vec[root_out_degrees >= 0]
            wid = n_repr['wid']
            p_inputs.append(torch.cat([x, h, tree_vec_set], 1))
            # Only newly generated nodes are needed for label prediction
            # NOTE: The following works since the uncomputed messages are zeros.

            q_input = torch.cat([h, tree_vec_set], 1)[is_new]
            q_target = wid[is_new]
            if q_input.shape[0] > 0:
                q_inputs.append(q_input)
                q_targets.append(q_target)
        p_targets.append(torch.zeros((root_out_degrees == 0).sum()).long())

        # Batch compute the stop/label prediction losses
        p_inputs = torch.cat(p_inputs, 0)
        p_targets = torch.cat(p_targets, 0).to(device)
        q_inputs = torch.cat(q_inputs, 0)
        q_targets = torch.cat(q_targets, 0)

        q = self.W_o(torch.relu(self.W(q_inputs)))
        p = self.U_s(torch.relu(self.U(p_inputs)))[:, 0]

        p_loss = F.binary_cross_entropy_with_logits(
            p, p_targets.float(), reduction='sum'
        ) / n_trees
        q_loss = F.cross_entropy(q, q_targets, reduction='sum') / n_trees
        p_acc = ((p > 0).long() == p_targets).sum().float() / \
            p_targets.shape[0]
        q_acc = (q.max(1)[1] == q_targets).float().sum() / q_targets.shape[0]

        self.q_inputs = q_inputs
        self.q_targets = q_targets
        self.q = q
        self.p_inputs = p_inputs
        self.p_targets = p_targets
        self.p = p

        return q_loss, p_loss, q_acc, p_acc

    def decode(self, mol_vec, max_decode_len=100):
        assert mol_vec.shape[0] == 1
        device = mol_vec.device

        mol_tree = DGLMolTree(None)
        mol_tree.g = mol_tree.g.to(device)

        init_hidden = torch.zeros(1, self.hidden_size).to(device)

        root_hidden = torch.cat([init_hidden, mol_vec], 1)
        root_hidden = F.relu(self.W(root_hidden))
        root_score = self.W_o(root_hidden)
        _, root_wid = torch.max(root_score, 1)
        root_wid = root_wid.view(1)

        mol_tree.g.add_nodes(1)   # root
        mol_tree.g.nodes[0].data['wid'] = root_wid
        mol_tree.g.nodes[0].data['x'] = self.embedding(root_wid)
        mol_tree.g.nodes[0].data['h'] = init_hidden
        mol_tree.g.nodes[0].data['fail'] = torch.tensor([0]).to(device)
        mol_tree.nodes_dict[0] = root_node_dict = create_node_dict(
            self.vocab.get_smiles(root_wid))

        stack, trace = [], []
        stack.append((0, self.vocab.get_slots(root_wid)))

        all_nodes = {0: root_node_dict}
        first = True
        new_node_id = 0
        new_edge_id = 0

        for step in range(max_decode_len):
            u, u_slots = stack[-1]
            udata = mol_tree.g.nodes[u].data
            x = udata['x']
            h = udata['h']

            # Predict stop
            p_input = torch.cat([x, h, mol_vec], 1)
            p_score = torch.sigmoid(self.U_s(torch.relu(self.U(p_input))))
            backtrack = (p_score.item() < 0.5)

            if not backtrack:
                # Predict next clique.  Note that the prediction may fail due
                # to lack of assemblable components
                mol_tree.g.add_nodes(1)
                new_node_id += 1
                v = new_node_id
                mol_tree.g.add_edges(u, v)
                uv = new_edge_id
                new_edge_id += 1

                if first:
                    mol_tree.g.edata.update({
                        's': torch.zeros(1, self.hidden_size).to(device),
                        'm': torch.zeros(1, self.hidden_size).to(device),
                        'r': torch.zeros(1, self.hidden_size).to(device),
                        'z': torch.zeros(1, self.hidden_size).to(device),
                        'src_x': torch.zeros(1, self.hidden_size).to(device),
                        'dst_x': torch.zeros(1, self.hidden_size).to(device),
                        'rm': torch.zeros(1, self.hidden_size).to(device),
                        'accum_rm': torch.zeros(1, self.hidden_size).to(device),
                    })
                    first = False

                mol_tree.g.edges[uv].data['src_x'] = mol_tree.g.nodes[u].data['x']
                # keeping dst_x 0 is fine as h on new edge doesn't depend on that.

                # DGL doesn't dynamically maintain a line graph.
                mol_tree_lg = dgl.line_graph(mol_tree.g, backtracking=False, shared=True)
                mol_tree_lg._node_frames = mol_tree.g._edge_frames

                mol_tree_lg.pull(
                    uv,
                    dec_tree_edge_msg1,
                    dec_tree_edge_reduce1
                )
                mol_tree_lg.pull(
                    uv,
                    dec_tree_edge_msg2,
                    dec_tree_edge_reduce2,
                    self.dec_tree_edge_update.update_zm
                )
                mol_tree.g.pull(
                    v,
                    dec_tree_node_msg,
                    dec_tree_node_reduce,
                )

                vdata = mol_tree.g.nodes[v].data
                h_v = vdata['h']
                q_input = torch.cat([h_v, mol_vec], 1)
                q_score = torch.softmax(
                    self.W_o(torch.relu(self.W(q_input))), -1)
                _, sort_wid = torch.sort(q_score, 1, descending=True)
                sort_wid = sort_wid.squeeze()

                next_wid = None
                for wid in sort_wid.tolist()[:5]:
                    slots = self.vocab.get_slots(wid)
                    cand_node_dict = create_node_dict(
                        self.vocab.get_smiles(wid))
                    if (have_slots(u_slots, slots) and can_assemble(mol_tree, u, cand_node_dict)):
                        next_wid = wid
                        next_slots = slots
                        next_node_dict = cand_node_dict
                        break

                if next_wid is None:
                    # Failed adding an actual children; v is a spurious node
                    # and we mark it.
                    vdata['fail'] = torch.tensor([1]).to(device)
                    backtrack = True
                else:
                    next_wid = torch.tensor([next_wid]).to(device)
                    vdata['wid'] = next_wid
                    vdata['x'] = self.embedding(next_wid)
                    mol_tree.nodes_dict[v] = next_node_dict
                    all_nodes[v] = next_node_dict
                    stack.append((v, next_slots))
                    mol_tree.g.add_edges(v, u)
                    vu = new_edge_id
                    new_edge_id += 1
                    mol_tree.g.edges[uv].data['dst_x'] = mol_tree.g.nodes[v].data['x']
                    mol_tree.g.edges[vu].data['src_x'] = mol_tree.g.nodes[v].data['x']
                    mol_tree.g.edges[vu].data['dst_x'] = mol_tree.g.nodes[u].data['x']

                    # DGL doesn't dynamically maintain a line graph.
                    mol_tree_lg = dgl.line_graph(mol_tree.g, backtracking=False, shared=True)
                    mol_tree_lg._node_frames = mol_tree.g._edge_frames
                    mol_tree_lg.apply_nodes(
                        self.dec_tree_edge_update.update_r,
                        uv
                    )

            if backtrack:
                if len(stack) == 1:
                    break   # At root, terminate

                pu, _ = stack[-2]
                u_pu = mol_tree.g.edge_ids(u, pu)

                mol_tree_lg.pull(
                    u_pu,
                    dec_tree_edge_msg1,
                    dec_tree_edge_reduce1
                )
                mol_tree_lg.pull(
                    u_pu,
                    dec_tree_edge_msg2,
                    dec_tree_edge_reduce2,
                    self.dec_tree_edge_update
                )
                mol_tree.g.pull(
                    pu,
                    dec_tree_node_msg,
                    dec_tree_node_reduce,
                )
                stack.pop()

        effective_nodes = mol_tree.g.filter_nodes(
            lambda nodes: nodes.data['fail'] != 1)
        effective_nodes, _ = torch.sort(effective_nodes)
        return mol_tree, all_nodes, effective_nodes
