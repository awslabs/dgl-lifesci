import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from torch.nn import init

__all__ = ['PotentialNet']

def sum_ligand_features(h, batch_num_nodes):
    """
    Compute the sum of only ligand features `h` according to the batch information `batch_num_nodes`.
    """
    node_nums = th.cumsum(batch_num_nodes, dim=0)
    B = int(len(batch_num_nodes) / 2) # actual batch size
    ligand_idx = [list(range(node_nums[0]))] # first ligand
    for i in range(2, len(node_nums), 2): # the rest of ligands in the batch
        ligand_idx.append(list(range(node_nums[i-1], node_nums[i])))
    return th.cat([h[i,].sum(0, keepdim=True) for i in ligand_idx]).to(device=h.device) # sum over each ligand


class PotentialNet(nn.Module):
    """
    Protein-ligand binding affinity prediction using a 'staged gated graph neural network'
    introduced in `PotentialNet for Molecular Property Prediction <http://dx.doi.org/10.1021/acscentsci.8b00507>`__.

    Parameters
    ----------
    f_in: int
        The dimension size of input features to GatedGraphConv, 
        equivalent to the dimension size of atomic features in the molecular graph.
    f_bond: int
        The dimension size of the output from GatedGraphConv in stage 1,
        equivalent to the dimension size of input to the linear layer at the end of stage 1.
    f_spatial: int
        The dimension size of the output from GatedGraphConv in stage 2,
        equivalent to the dimension size of input to the linear layer at the end of stage 2.
    f_gather: int
        The dimension size of the output from stage 1 & 2,
        equivalent to the dimension size of output from the linear layer at the end of stage 1 & 2.
    n_etypes: int
        The number of heterogeneous edge types for stage 2. This includes the number of covalent bond types from stage 1 and the number of spatial edge types based on distances.
        Default to 9 (5 covalent bond types in stage 1 plus 4 distance bins in stage 2).
    n_bond_conv_steps: int
        The number of bond convolution layers(steps) of GatedGraphConv in stage 1.
    n_spatial_conv_steps: int
        The number of spatial convolution layers(steps) of GatedGraphConv in stage 2.
    n_rows_fc: list of int
        The widths of the fully connected neural networks at each layer in stage 3.
    dropouts: list of 3 floats
        The amount of dropout applied at the end of each stage.
    """
    def __init__(self,
                 f_in,
                 f_bond,
                 f_spatial,
                 f_gather,
                 n_etypes,
                 n_bond_conv_steps,
                 n_spatial_conv_steps,
                 n_rows_fc,
                 dropouts
                 ):
        super(PotentialNet, self).__init__()
        self.stage_1_model = CustomizedGatedGraphConv(in_feats=f_in,
                                                      out_feats=f_bond,
                                                      f_gather=f_gather,
                                                      n_etypes=5,
                                                      n_steps=n_bond_conv_steps,
                                                      dropout=dropouts[0])
        self.stage_2_model = CustomizedGatedGraphConv(in_feats=f_gather,
                                                      out_feats=f_spatial,
                                                      f_gather=f_gather,
                                                      n_etypes=n_etypes, # num_distance_bins + 5 covalent types
                                                      n_steps=n_spatial_conv_steps,
                                                      dropout=dropouts[1])
        self.stage_3_model = StagedFCNN(f_in=f_gather,
                                        n_row=n_rows_fc,
                                        dropout=dropouts[2])

    def forward(self, bigraph, knn_graph):
        """
        Compute the prediction on graphs using PotentialNet model.

        Parameters
        ----------
        bigraph: DGLGraph
            The molecular graph for stage 1 of PotentialNet, with `bigraph.ndata['h']` being the input node features.
            and `bigraph.edata['e']` being the one-hot encoding of the edge types.
        knn_graph: DGLGraph
            The k-nearest-neighbor graph for stage 2 of PotentialNet, with no initial node features
            and `knn_graph.edata['e']` being the one-hot encoding of the edge types.

        Returns
        -------
        x: torch.Tensor
            The prediction based on the input features and graphs.
            For the task of binding affinity prediction, the shape is (B, 1), where B is the batch size.
        """
        batch_num_nodes = bigraph.batch_num_nodes()
        h = self.stage_1_model(graph=bigraph, feat=bigraph.ndata['h'])
        h = self.stage_2_model(graph=knn_graph, feat=h)
        x = self.stage_3_model(batch_num_nodes=batch_num_nodes, features=h) 
        return x

class StagedFCNN(nn.Module):
    """
    The implementation of PotentialNet stage 3.
    A graph gather is performed solely on the ligand atoms followed by a multi-layer fully connected neural network.

    Parameters
    ----------
    f_in: int
        The input feature size.
    n_row: list of int
        The widths of a sequence of linear layers.
        The number of layers will be the length of the list plus 1.
    dropout: float
        Dropout to be applied before each layer, except the first.
    """
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
        """
        Gather features on ligands and compute with fully connected linear layers.

        Parameters
        ----------
        batch_num_nodes: torch.Tensor
            The number of nodes for each graph in the batch as from `DGLGraph.batch_num_nodes()`.
        features: torch.Tensor
            Node features from the output of GatedGraphConv.
        """
        x = sum_ligand_features(features, batch_num_nodes)
        for i, layer in enumerate(self.layers):
            if i !=0:
                x = self.dropout(x)
            x = layer(x)
            x = F.relu(x)
        x = self.out_layer(x)
        return x

class CustomizedGatedGraphConv(nn.Module):
    """
    Adapted from `dgl.nn.pytorch.conv.GatedGraphConv`.
    Customized the implementation for applying edges for better performance.
    Added a linear layer at the end as described in PotentialNet stage 1 & 2.
    
    Parameters
    ----------
    in_feats: int
        Input feature size.
    out_feats: int
        Output feature size from GatedGraphConv,
        equivalent to the input feature size to the linear layer.
    f_gather: int
        Output feature size from the linear layer.
    n_steps: int
        Number of recurrent steps.
    n_etypes: int
        Number of edge types.
    dropout: float
        Amount of dropout applied between the GatedGraphConv module and the linear layer.
    bias: bool
        If True, adds a learnable bias to the output. Default to True.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 f_gather,
                 n_steps,
                 n_etypes,
                 dropout,
                 bias=True):
        super(CustomizedGatedGraphConv, self).__init__()
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


    def forward(self, graph, feat):
        """
        Description
        -----------
        Compute Gated Graph Convolution layer.

        Parameters
        ----------
        graph: DGLGraph
            The graph to run gated graph convolution,
            with graph.edata['e'] being the one-hot encodings of the edge types.
        feat: torch.Tensor
            The input feature as the node features in `graph`.
            Dimension: (N, `self._in_feats`), where N is the number of nodes in `graph`.

        Returns
        -------
        torch.Tensor
            The output feature of dimension (N, `self._out_feats`).
        """
        with graph.local_scope():
            assert graph.is_homogeneous, \
                "not a homogeneous graph; convert it with to_homogeneous " \
                "and pass in the edge type as argument"
            assert graph.edata['e'].shape[1] <= self._n_etypes, \
                "edge type indices out of range [0, {})".format(self._n_etypes)
            zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
            h = th.cat([feat, zero_pad], -1)

            for _ in range(self._n_steps):
                graph.ndata['h'] = h
                for i in range(self._n_etypes):
                    eids = graph.edata['e'][:,i].nonzero(as_tuple=False).view(-1).type(graph.idtype)
                    if len(eids) > 0:
                        graph.apply_edges(
                            lambda edges: {'W_e*h': self.linears[i](edges.src['h'])},
                            eids
                        )
                graph.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
                a = graph.ndata.pop('a') # (N, D)
                h = self.gru(a, h)

            h = self.dropout(h)
            h = th.mul(
                    th.sigmoid(self.i_nn(th.cat((h, feat),dim=1))), 
                    self.j_nn(h))    
            return h
