import math
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch.conv import GatedGraphConv
from torch.nn import init

__all__ = ['PotentialNet']

def process_etypes(graph):
    """convert one-hot encoding edge types to label encoding, and add duplicated edges
    """
    edata = graph.edata['e']
    etypes = th.tensor([], dtype=th.long, device=graph.device)
    for i in range(edata.shape[0]):
        encodings = th.nonzero(edata[i,])
        etypes = th.cat([etypes, encodings[0]])
        src, dst = graph.find_edges(i)
        encodings = encodings.view(-1)
        num_2_add = encodings[1:].shape # start from the second
        # add edges repeatedly to represent different edge types
        graph.add_edges(src.repeat(num_2_add), dst.repeat(num_2_add)) 
        etypes = th.cat([etypes, encodings[1:]])
    del graph.edata['e']
    return graph, etypes

def sum_ligand_features(h, batch_num_nodes):
    """Computes the sum of ligand features h from batch_num_nodes"""
    node_nums = th.cumsum(batch_num_nodes, dim=0)
    B = int(len(batch_num_nodes)/2) # actual batch size
    ligand_idx = [list(range(node_nums[0]))] # first ligand
    for i in range(2,len(node_nums),2): # the rest of ligands in the batch
        ligand_idx.append(list(range(node_nums[i-1],node_nums[i])))
    return th.cat([h[i,].sum(0, keepdim=True) for i in ligand_idx]).to(device=h.device) # sum over each ligand


class PotentialNet(nn.Module):
    def __init__(self,
                 n_etypes,
                 f_in,
                 f_bond,
                 f_spatial,
                 f_gather,
                 n_rows_fc,
                 n_bond_conv_steps,
                 n_spatial_conv_steps,
                 dropouts
                 ):
        super(PotentialNet, self).__init__()
        self.stage_1_model = Customized_GatedGraphConv(in_feats=f_in,
                                        out_feats=f_bond,
                                        f_gather=f_gather,
                                        n_etypes=5,
                                        n_steps=n_bond_conv_steps,
                                        dropout=dropouts[0]
                                        )
        self.stage_2_model = Customized_GatedGraphConv(in_feats=f_gather,
                                        out_feats=f_spatial,
                                        f_gather=f_gather,
                                        n_etypes=n_etypes, # num_distance_bins + 5 covalent types
                                        n_steps=n_spatial_conv_steps,
                                        dropout=dropouts[1]
                                        )
        self.stage_3_model = StagedFCNN(f_in=f_gather,
                                        n_row=n_rows_fc,
                                        dropout=dropouts[2]
        )

    def forward(self, bigraph, knn_graph):
        batch_num_nodes = bigraph.batch_num_nodes()
        # bigraph, stage_1_etypes = process_etypes(bigraph_canonical)
        h = self.stage_1_model(graph=bigraph, feat=bigraph.ndata['h'])
        h = self.stage_2_model(graph=knn_graph, feat=h)
        x = self.stage_3_model(batch_num_nodes=batch_num_nodes, features=h) 
        return x


class StagedGGNN(nn.Module):
    def __init__(self,
                 f_gru_in,
                 f_gru_out,
                 f_gather,
                 n_etypes,
                 n_gconv_steps,
                 dropout):
        super(StagedGGNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.ggc = GatedGraphConv(in_feats=f_gru_in, 
                                      out_feats=f_gru_out,
                                      n_steps=n_gconv_steps,
                                      n_etypes=n_etypes,
                                      bias=True
                                      )
        self.i_nn = nn.Linear(in_features=(f_gru_out + f_gru_in), out_features=f_gather)
        self.j_nn = nn.Linear(in_features=f_gru_out, out_features=f_gather)

    def forward(self, graph, features, etypes):
        # h = features
        h = self.ggc(graph, features, etypes)
        h = self.dropout(h) # my guess
        h = th.mul(
                   th.sigmoid(self.i_nn(th.cat((h, features),dim=1))), 
                   self.j_nn(h))
        return h

class StagedFCNN(nn.Module):
    def __init__(self,
                 f_in,
                 n_row,
                 dropout):
        super(StagedFCNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(f_in, n_row[0]))
        for i in range(1, len(n_row)):
            self.layers.append(nn.Linear(n_row[i-1], n_row[i]))
        self.out_layer = nn.Linear(n_row[-1], 1)

    def forward(self, batch_num_nodes, features):
        x = sum_ligand_features(features, batch_num_nodes)
        for i, layer in enumerate(self.layers):
            if i !=0:
                x = self.dropout(x)
            x = layer(x)
            x = F.relu(x)
        x = self.out_layer(x)
        return x


class Customized_GatedGraphConv(nn.Module):
    def __init__(self,
                in_feats,
                out_feats,
                f_gather,
                n_steps,
                n_etypes,
                dropout,
                bias=True):
        super(Customized_GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.i_nn = nn.Linear(in_features=(in_feats + out_feats), out_features=f_gather)
        self.j_nn = nn.Linear(in_features=out_feats, out_features=f_gather)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        """

        Description
        -----------
        Compute Gated Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph, with graph.ndata['h'] being the input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size;
            And graph.edata['e'] one-hot encodes the edge types

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            assert graph.is_homogeneous, \
                "not a homogeneous graph; convert it with to_homogeneous " \
                "and pass in the edge type as argument"
            assert  graph.edata['e'].shape[1] <= self._n_etypes, \
                "edge type indices out of range [0, {})".format(self._n_etypes)
            zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
            h = th.cat([feat, zero_pad], -1)

            for _ in range(self._n_steps):
                graph.ndata['h'] = h
                for i in range(self._n_etypes):
                    eids = graph.edata['e'][:,i].nonzero().view(-1).type(graph.idtype)
                    if len(eids) > 0:
                        graph.apply_edges(
                            lambda edges: {'W_e*h': self.linears[i](edges.src['h'])},
                            eids
                        )
                graph.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
                a = graph.ndata.pop('a') # (N, D)
                h = self.gru(a, h)

            h = self.dropout(h) # my guess
            h = th.mul(
                    th.sigmoid(self.i_nn(th.cat((h, feat),dim=1))), 
                    self.j_nn(h))    
            return h
