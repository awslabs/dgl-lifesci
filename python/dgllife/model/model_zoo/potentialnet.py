import numpy as np
import math
import torch as th
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GatedGraphConv
import torch.nn.functional as F

__all__ = ['PotentialNet']

def process_etypes(graph):
    """convert one-hot encoding edge types to label encoding, and add duplicated edges
    """
    edata = graph.edata['e']
    etypes = th.tensor([], dtype=th.long)
    for i in range(edata.shape[0]):
        print(etypes.shape)
        encodings = th.nonzero(edata[i,])
        etypes = th.cat([etypes, encodings[0]])
        src, dst = graph.find_edges(i)
        for _ in encodings[1:]: # start from the second
        # add edges repeatedly to represent different edge types
            graph.add_edges(src, dst)
        etypes = th.cat([etypes, encodings[1:].reshape(1,-1)[0]])
    return graph, etypes

def sum_ligand_features(h, batch_num_nodes):
    """Computes the sum of ligand features h from batch_num_nodes"""
    node_nums = th.cumsum(batch_num_nodes, dim=0)
    B = int(len(batch_num_nodes)/2)
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
                 n_row_fc,
                 n_bond_conv_steps,
                 n_spatial_conv_steps,
                 n_fc_layers,
                 dropout
                 ):
        super(PotentialNet, self).__init__()
        self.stage_1_model = StagedGGNN(f_gru_in=f_in,
                                        f_gru_out=f_bond,
                                        f_gather=f_gather,
                                        n_etypes=12, # from CanonicalBondFeaturizer
                                        n_gconv_steps=n_bond_conv_steps,
                                        dropout=dropout
                                        )
        self.stage_2_model = StagedGGNN(f_gru_in=f_gather,
                                        f_gru_out=f_spatial,
                                        f_gather=f_gather,
                                        n_etypes=n_etypes, # all 
                                        n_gconv_steps=n_spatial_conv_steps,
                                        dropout=dropout
                                        )
        self.stage_3_model = StagedFCNN(f_in=f_gather,
                                        n_row=n_row_fc,
                                        n_layers=n_fc_layers,
                                        dropout=dropout
        )

    def forward(self, bigraph_canonical, knn_graph):
        batch_num_nodes = bigraph_canonical.batch_num_nodes()
        bigraph, stage_1_etypes = process_etypes(bigraph_canonical)
        stage_2_etypes = th.zeros(knn_graph.num_edges() ,dtype=th.long)   # temporal solution
        h = self.stage_1_model(graph=bigraph, features=bigraph.ndata['h'], etypes=stage_1_etypes)
        h = self.stage_2_model(graph=knn_graph, features=h, etypes=stage_2_etypes)
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
                 n_layers,
                 dropout):
        super(StagedFCNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(f_in, n_row))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_row, n_row))
        self.out_layer = nn.Linear(n_row, 1)

    def forward(self, batch_num_nodes, features):
        x = sum_ligand_features(features, batch_num_nodes)
        for i, layer in enumerate(self.layers):
            if i !=0:
                x = self.dropout(x)
            x = layer(x)
            x = F.relu(x)
        x = self.out_layer(x)
        return x

