import torch.nn as nn

from dgl.nn.pytorch import Set2Set
from dgllife.model import MPNNGNN

from .regressor import BaseGNNRegressor, BaseGNNRegressorBypass

class MPNNRegressor(BaseGNNRegressor):
    """
    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    node_hidden_dim : int
        Size for hidden node representations
    edge_hidden_dim : int
        Size for hidden edge representations
    num_step_message_passing : int
        Number of message passing steps
    num_step_set2set : int
        Number of Set2Set steps
    num_layer_set2set : int
        Number of Set2Set layers
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_node_feats, in_edge_feats, node_hidden_dim, edge_hidden_dim,
                 num_step_message_passing, num_step_set2set, num_layer_set2set, n_tasks,
                 regressor_hidden_feats=128, dropout=0.):
        super(MPNNRegressor, self).__init__(readout_feats= 2 * node_hidden_dim,
                                            n_tasks=n_tasks,
                                            regressor_hidden_feats=regressor_hidden_feats,
                                            dropout=dropout)
        self.gnn = MPNNGNN(in_node_feats, in_edge_feats, node_hidden_dim,
                           edge_hidden_dim, num_step_message_passing)
        self.readout = Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)

class MPNNRegressorBypass(BaseGNNRegressorBypass):
    """MPNN-based model for bypass multitask molecular property prediction.
    We assume all tasks are regression problems.

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    node_hidden_dim : int
        Size for hidden node representations
    edge_hidden_dim : int
        Size for hidden edge representations
    num_step_message_passing : int
        Number of message passing steps
    num_step_set2set : int
        Number of Set2Set steps
    num_layer_set2set : int
        Number of Set2Set layers
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_node_feats, in_edge_feats, node_hidden_dim, edge_hidden_dim,
                 num_step_message_passing, num_step_set2set, num_layer_set2set, n_tasks,
                 regressor_hidden_feats=128, dropout=0.):
        super(MPNNRegressorBypass, self).__init__(
            readout_feats= 4 * node_hidden_dim, n_tasks=n_tasks,
            regressor_hidden_feats=regressor_hidden_feats,
            dropout=dropout)
        self.shared_gnn = MPNNGNN(in_node_feats, in_edge_feats, node_hidden_dim,
                                  edge_hidden_dim, num_step_message_passing)
        for _ in range(n_tasks):
            self.task_gnns.append(MPNNGNN(in_node_feats, in_edge_feats, node_hidden_dim,
                                          edge_hidden_dim, num_step_message_passing))
            self.readouts.append(Set2Set(2 * node_hidden_dim, num_step_set2set, num_layer_set2set))
