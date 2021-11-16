import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    """MLP for regression (over multiple tasks) from molecule representations.

    Parameters
    ----------
    in_feats : int
        Number of input molecular graph features
    hidden_feats : int
        Hidden size for molecular graph representations
    n_tasks : int
        Number of tasks, also output size
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(MLPRegressor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, h):
        """Predict for regression.

        Parameters
        ----------
        h : FloatTensor of shape (B, M3)
            * B is the number of molecules in a batch
            * M3 is the input molecule feature size, must match in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
        """
        return self.predict(h)

class BaseGNNRegressor(nn.Module):
    """GNN based model for multitask molecular property prediction.
    We assume all tasks are regression problems.

    Parameters
    ----------
    readout_feats : int
        Size for molecular representations
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, readout_feats, n_tasks, regressor_hidden_feats=128, dropout=0.):
        super(BaseGNNRegressor, self).__init__()

        self.regressor = MLPRegressor(readout_feats, regressor_hidden_feats, n_tasks, dropout)

    def forward(self, bg, node_feats, edge_feats):
        """Multi-task prediction for a batch of molecules

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
        FloatTensor of shape (B, n_tasks)
            Prediction for all tasks on the batch of molecules
        """
        # Update node representations
        feats = self.gnn(bg, node_feats, edge_feats)

        # Compute molecule features from atom features
        h_g = self.readout(bg, feats)

        # Multi-task prediction
        return self.regressor(h_g)

class BaseGNNRegressorBypass(nn.Module):
    """This architecture uses one GNN for each task (task-speicifc) and one additional GNN shared
    across all tasks. To predict for each task, we feed the input to both the task-specific GNN
    and the task-shared GNN. The resulted representations of the two GNNs are then concatenated
    and fed to a task-specific forward NN.

    Parameters
    ----------
    readout_feats : int
        Size for molecular representations
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, readout_feats, n_tasks, regressor_hidden_feats=128, dropout=0.):
        super(BaseGNNRegressorBypass, self).__init__()

        self.n_tasks = n_tasks
        self.task_gnns = nn.ModuleList()
        self.readouts = nn.ModuleList()
        self.regressors = nn.ModuleList()

        for _ in range(n_tasks):
            self.regressors.append(
                MLPRegressor(readout_feats, regressor_hidden_feats, 1, dropout))

    def forward(self, bg, node_feats, edge_feats):
        """Multi-task prediction for a batch of molecules

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
        FloatTensor of shape (B, n_tasks)
            Prediction for all tasks on the batch of molecules
        """
        shared_repr = self.shared_gnn(bg, node_feats, edge_feats)
        predictions = []

        for t in range(self.n_tasks):
            task_repr = self.task_gnns[t](bg, node_feats, edge_feats)
            combined_repr = torch.cat([shared_repr, task_repr], dim=1)
            g_t = self.readouts[t](bg, combined_repr)
            predictions.append(self.regressors[t](g_t))

        # Combined predictions of all tasks
        return torch.cat(predictions, dim=1)
