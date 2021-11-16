import torch.nn as nn

from dgllife.model import GCN, WeightedSumAndMax

from .blocks import InputInitializer, EdgeGraphConv
from .regressor import BaseGNNRegressor, BaseGNNRegressorBypass

class GNN(nn.Module):
    """A GCN variant where we combine the node and edge features in the first layer.

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    gcn_hidden_feats : list[int]
        gcn_hidden_feats[i] gives the output representation size in the i+1-th gcn layer
    """
    def __init__(self, in_node_feats, in_edge_feats, gcn_hidden_feats):
        super(GNN, self).__init__()

        self.e_repr_initializer = InputInitializer(in_node_feats, in_edge_feats)
        self.edge_conv = EdgeGraphConv(in_node_feats + in_edge_feats, gcn_hidden_feats[0])
        self.gcn = GCN(in_feats=gcn_hidden_feats[0],
                       hidden_feats=gcn_hidden_feats[1:])

    def forward(self, bg, node_feats, edge_feats):
        """Update node representations.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of B graphs
        node_feats : FloatTensor of shape (N, D0)
            Initial features for all nodes in the batch of graphs
        edge_feats : FloatTensor of shape (M, D1)
            Initial features for all edges in the batch of graphs

        Returns
        -------
        feats : FloatTensor of shape (N, gcn_hidden_feats[-1])
            Updated node representations
        """
        # Initialize edge representations.
        feats = self.e_repr_initializer(bg, node_feats, edge_feats)
        feats = self.edge_conv(bg, feats)

        return self.gcn(bg, feats)

class GCNRegressor(BaseGNNRegressor):
    """GCN-based model for multitask molecular property prediction.
    We assume all tasks are regression problems.

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    gcn_hidden_feats : list[int]
        gcn_hidden_feats[i] gives the output representation size in the i+1-th gcn layer
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_node_feats, in_edge_feats, gcn_hidden_feats, n_tasks,
                 regressor_hidden_feats=128, dropout=0.):
        super(GCNRegressor, self).__init__(readout_feats= 2 * gcn_hidden_feats[-1],
                                           n_tasks=n_tasks,
                                           regressor_hidden_feats=regressor_hidden_feats,
                                           dropout=dropout)

        self.gnn = GNN(in_node_feats, in_edge_feats, gcn_hidden_feats)
        self.readout = WeightedSumAndMax(gcn_hidden_feats[-1])

class GCNRegressorBypass(BaseGNNRegressorBypass):
    """GCN-based model for bypass multitask molecular property prediction.
    We assume all tasks are regression problems.

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    gcn_hidden_feats : list[int]
        gcn_hidden_feats[i] gives the output representation size in the i+1-th gcn layer
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_node_feats, in_edge_feats, gcn_hidden_feats, n_tasks,
                 regressor_hidden_feats=128, dropout=0.):
        super(GCNRegressorBypass, self).__init__(readout_feats= 4 * gcn_hidden_feats[-1],
                                                 n_tasks=n_tasks,
                                                 regressor_hidden_feats=regressor_hidden_feats,
                                                 dropout=dropout)
        self.shared_gnn = GNN(in_node_feats, in_edge_feats, gcn_hidden_feats)
        for _ in range(n_tasks):
            self.task_gnns.append(GNN(in_node_feats, in_edge_feats, gcn_hidden_feats))
            self.readouts.append(WeightedSumAndMax(2 * gcn_hidden_feats[-1]))
