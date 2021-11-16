import torch.nn as nn

from dgllife.model import AttentiveFPGNN, AttentiveFPReadout

from .regressor import BaseGNNRegressor, BaseGNNRegressorBypass

class AttentiveFPRegressor(BaseGNNRegressor):
    """AttentiveFP-based model for multitask molecular property prediction.
    We assume all tasks are regression problems.

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    gnn_out_feats : int
        The GNN output size
    num_layers : int
        Number of GNN layers
    num_timesteps : int
        Number of timesteps for updating molecular representations with GRU during readout
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_node_feats, in_edge_feats, gnn_out_feats, num_layers, num_timesteps,
                 n_tasks, regressor_hidden_feats=128, dropout=0.):
        super(AttentiveFPRegressor, self).__init__(readout_feats=gnn_out_feats,
                                                   n_tasks=n_tasks,
                                                   regressor_hidden_feats=regressor_hidden_feats,
                                                   dropout=dropout)
        self.gnn = AttentiveFPGNN(in_node_feats, in_edge_feats, num_layers,
                                  gnn_out_feats, dropout)
        self.readout = AttentiveFPReadout(gnn_out_feats, num_timesteps, dropout)

class AttentiveFPRegressorBypass(BaseGNNRegressorBypass):
    """AttentiveFP-based model for bypass multitask molecular property prediction.
    We assume all tasks are regression problems.

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    gnn_out_feats : int
        The GNN output size
    num_layers : int
        Number of GNN layers
    num_timesteps : int
        Number of timesteps for updating molecular representations with GRU during readout
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_node_feats, in_edge_feats, gnn_out_feats, num_layers, num_timesteps,
                 n_tasks, regressor_hidden_feats=128, dropout=0.):
        super(AttentiveFPRegressorBypass, self).__init__(
            readout_feats= 2 * gnn_out_feats, n_tasks=n_tasks,
            regressor_hidden_feats=regressor_hidden_feats,
            dropout=dropout)
        self.shared_gnn = AttentiveFPGNN(in_node_feats, in_edge_feats, num_layers,
                                         gnn_out_feats, dropout)
        for _ in range(n_tasks):
            self.task_gnns.append(AttentiveFPGNN(in_node_feats, in_edge_feats, num_layers,
                                                 gnn_out_feats, dropout))
            self.readouts.append(AttentiveFPReadout(2 * gnn_out_feats, num_timesteps, dropout))
