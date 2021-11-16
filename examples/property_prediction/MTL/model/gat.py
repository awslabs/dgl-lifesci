import torch.nn as nn

from dgllife.model import GAT, WeightedSumAndMax

from .blocks import InputInitializer, EdgeGraphConv
from .regressor import BaseGNNRegressor, BaseGNNRegressorBypass

class GNN(nn.Module):
    """A GAT variant where we combine the node and edge features in the first layer.

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    gat_hidden_feats : list[int]
        gat_hidden_feats[i] gives the size for hidden representations in each head of
        the i-th attention layer.
    num_heads : list[int]
        num_heads[i] gives the number of attention heads in the i-th attention layer.
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_node_feats, in_edge_feats, gat_hidden_feats, num_heads, dropout=0.):
        super(GNN, self).__init__()

        self.e_repr_initializer = InputInitializer(in_node_feats, in_edge_feats)
        self.edge_conv = EdgeGraphConv(in_node_feats + in_edge_feats, in_node_feats + in_edge_feats)
        num_gat_layers = len(gat_hidden_feats)
        self.gat = GAT(in_feats=in_node_feats + in_edge_feats,
                       hidden_feats=gat_hidden_feats,
                       num_heads=num_heads,
                       feat_drops=[dropout] * num_gat_layers,
                       attn_drops=[dropout] * num_gat_layers)

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
        feats : FloatTensor of shape (N, gat_hidden_feats[-1])
            Updated node representations
        """
        # Initialize edge representations.
        feats = self.e_repr_initializer(bg, node_feats, edge_feats)
        feats = self.edge_conv(bg, feats)

        return self.gat(bg, feats)

class GATRegressor(BaseGNNRegressor):
    """GAT-based model for multitask molecular property prediction.
    We assume all tasks are regression problems.

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    gat_hidden_feats : list[int]
        gat_hidden_feats[i] gives the size for hidden representations in each head of
        the i-th attention layer.
    num_heads : list[int]
        num_heads[i] gives the number of attention heads in the i-th attention layer.
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_node_feats, in_edge_feats, gat_hidden_feats, num_heads, n_tasks,
                 regressor_hidden_feats=128, dropout=0.):
        super(GATRegressor, self).__init__(readout_feats= 2 * gat_hidden_feats[-1],
                                           n_tasks=n_tasks,
                                           regressor_hidden_feats=regressor_hidden_feats,
                                           dropout=dropout)

        self.gnn = GNN(in_node_feats, in_edge_feats, gat_hidden_feats, num_heads, dropout)
        self.readout = WeightedSumAndMax(gat_hidden_feats[-1])

class GATRegressorBypass(BaseGNNRegressorBypass):
    """GAT-based model for bypass multitask molecular property prediction.
    We assume all tasks are regression problems.

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    gat_hidden_feats : list[int]
        gat_hidden_feats[i] gives the size for hidden representations in each head of
        the i-th attention layer.
    num_heads : list[int]
        num_heads[i] gives the number of attention heads in the i-th attention layer.
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_node_feats, in_edge_feats, gat_hidden_feats, num_heads, n_tasks,
                 regressor_hidden_feats=128, dropout=0.):
        super(GATRegressorBypass, self).__init__(readout_feats= 4 * gat_hidden_feats[-1],
                                                 n_tasks=n_tasks,
                                                 regressor_hidden_feats=regressor_hidden_feats,
                                                 dropout=dropout)
        self.shared_gnn = GNN(in_node_feats, in_edge_feats, gat_hidden_feats, num_heads, dropout)
        for _ in range(n_tasks):
            self.task_gnns.append(GNN(in_node_feats, in_edge_feats,
                                      gat_hidden_feats, num_heads, dropout))
            self.readouts.append(WeightedSumAndMax(2 * gat_hidden_feats[-1]))
