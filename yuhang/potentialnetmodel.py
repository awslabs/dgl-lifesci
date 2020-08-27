import numpy as np
import os
import math
# import matplotlib.pyplot as plt
# import networkx as nx
import torch as th
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GatedGraphConv
import torch.nn.functional as F

class PotentialNet(nn.Module):
    def __init__(self,
                 bigraph, knn_graph, liagnd_graph
                 n_etypes,
                 f_in,
                 f_bonds,
                 f_spatial,
                 f_gather,
                 f_out,
                 n_row_fc,
                 n_bond_conv_steps,
                 n_spatial_conv_steps,
                 n_fc_layers,
                 f_fc,
                 lr,
                 dropout,
                 weight_decay,
                 ):
        super(PotentialNet, self).__init__()
        self.stage_1_model = StagedGGNN(f_gru_in=f_in,
                                        f_gru_out=f_bond,
                                        f_gather=f_gather,
                                        n_etypes=?, # distinghuish diff chem bonds
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
                                        f_out=f_out,
                                        n_layers=n_fc_layers,
                                        dropout=dropout
        )

    def forward(self, graph, features, stage_1_etypes, stage_2_etypes):
        # h = features
        h = self.stage_1_model(graph=bigraph, features=features, etypes=stage_1_etypes)
        h = self.stage_2_model(graph=knn_grpah, features=h, etypes=stage_2_etypes)
        x = self.stage_3_model(graph=ligand_graph, features=?)
        return h

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
        self.ggc= GatedGraphConv(in_feats=f_gru_in, 
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
                 f_out,
                 n_layers,
                 dropout):
        super(StagedFCNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(f_in, n_row))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_row, n_row))
        self.out_layer = nn.Linear(n_row, f_out)

    def forward(self, graph, features):
        x = sum_over_ligand(graph, features) # need to be defined, x is of dimension (1xf_gather)
        for i, layer in enumerate(self.layers):
            if i !=0:
                x = self.dropout(x)
            x = layer(x)
            x = F.relu(x)
        x = self.out_layer(x)
        return x
