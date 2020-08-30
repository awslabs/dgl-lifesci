import numpy as np
import math
import torch as th
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GatedGraphConv
import torch.nn.functional as F

__all__ = ['PotentialNet']

def process_etypes(graph):
    '''convert one-hot encoding edge types to label encoding, and add duplicated edges
    '''
    edata = np.array(graph.edata['e'])
    etypes = []
    etypes_to_extend = []
    for i in range(edata.shape[0]):
        row = edata[i,]
        encodings = np.nonzero(row)[0]
        etypes.append(encodings[0])
        src, dst = graph.find_edges(i)
        for _ in encodings[1:]: # start from the second
        # add edges to represent different edge types
            graph.add_edges(np.array(src), np.array(dst))
        etypes_to_extend.extend(encodings[1:])
    etypes.extend(etypes_to_extend)
    del graph.edata['e']
    return graph, th.tensor(etypes, dtype=th.long)

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
        num_atoms_ligand = int(bigraph_canonical.batch_num_nodes()[0])
        bigraph, stage_1_etypes = process_etypes(bigraph_canonical)
        stage_2_etypes = th.zeros(knn_graph.num_edges() ,dtype=th.long)   # temporal solution
        h = self.stage_1_model(graph=bigraph, features=bigraph.ndata['h'], etypes=stage_1_etypes)
        h = self.stage_2_model(graph=knn_graph, features=h, etypes=stage_2_etypes)
        x = self.stage_3_model(features=h[:num_atoms_ligand, ])
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
                   F.sigmoid(self.i_nn(th.cat((h, features),dim=1))), 
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

    def forward(self, features):
        x = features.sum(axis=0) # sum over ligand atoms
        for i, layer in enumerate(self.layers):
            if i !=0:
                x = self.dropout(x)
            x = layer(x)
            x = F.relu(x)
        x = self.out_layer(x)
        return x

