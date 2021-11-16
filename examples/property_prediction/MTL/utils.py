import dgl
import errno
import numpy as np
import os
import random
import torch

from model import *

def mkdir_p(path):
    """Create a folder for the given path.
    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def setup(args, random_seed=0):
    """Decide the device to use for computing, set random seed and perform sanity check."""
    if torch.cuda.is_available():
        args['device'] = 'cuda:0'
    else:
        args['device'] = 'cpu'

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Disable random behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if len(args['tasks']) == 1:
        assert args['mode'] == 'parallel', \
            'Bypass architecture is not applicable for single-task experiments.'

    return args

def get_label_mean_and_std(dataset):
    """Compute the mean and std of labels.

    Non-existing labels are excluded for computing mean and std.

    Parameters
    ----------
    dataset
        We assume that len(dataset) gives the number of datapoints
        in the dataset and dataset[i] gives the SMILES, RDKit molecule
        instance, DGLGraph, label and mask for the i-th datapoint.

    Returns
    -------
    labels_mean: float32 tensor of shape (T)
        Mean of the labels for all tasks
    labels_std: float32 tensor of shape (T)
        Std of the labels for all tasks
    """
    _, _, label, _ = dataset[0]
    n_tasks = label.shape[-1]
    task_values = {t: [] for t in range(n_tasks)}
    for i in range(len(dataset)):
        _, _, _, label, mask = dataset[i]
        for t in range(n_tasks):
            if mask[t].data.item() == 1.:
                task_values[t].append(label[t].data.item())

    labels_mean = torch.zeros(n_tasks)
    labels_std = torch.zeros(n_tasks)
    for t in range(n_tasks):
        labels_mean[t] = float(np.mean(task_values[t]))
        labels_std[t] = float(np.std(task_values[t]))

    return labels_mean, labels_std

def collate(data):
    """Batching a list of datapoints for dataloader in training GNNs.

    Returns
    -------
    smiles: list
        List of smiles
    bg: DGLGraph
        DGLGraph for a batch of graphs
    labels: Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks: Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))
    bg = dgl.batch(graphs)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

def load_model(args, node_featurizer, edge_featurizer):
    in_node_feats = node_featurizer.feat_size('hv')
    in_edge_feats = edge_featurizer.feat_size('he')

    if args['model'] == 'GCN':
        if args['mode'] == 'parallel':
            model_class = GCNRegressor
        else:
            model_class = GCNRegressorBypass
        model = model_class(in_node_feats=in_node_feats,
                            in_edge_feats=in_edge_feats,
                            gcn_hidden_feats=[args['gnn_hidden_feats']
                                              for _ in range(args['num_gnn_layers'])],
                            n_tasks=len(args['tasks']),
                            regressor_hidden_feats=args['regressor_hidden_feats'],
                            dropout=args['dropout'])
    elif args['model'] == 'GAT':
        if args['mode'] == 'parallel':
            model_class = GATRegressor
        else:
            model_class = GATRegressorBypass
        model = model_class(in_node_feats=in_node_feats,
                            in_edge_feats=in_edge_feats,
                            gat_hidden_feats=[args['gnn_hidden_feats']
                                              for _ in range(args['num_gnn_layers'])],
                            num_heads=[args['num_heads'] for _ in range(args['num_gnn_layers'])],
                            n_tasks=len(args['tasks']),
                            regressor_hidden_feats=args['regressor_hidden_feats'],
                            dropout=args['dropout'])
    elif args['model'] == 'MPNN':
        if args['mode'] == 'parallel':
            model_class = MPNNRegressor
        else:
            model_class = MPNNRegressorBypass
        model = model_class(in_node_feats=in_node_feats,
                            in_edge_feats=in_edge_feats,
                            node_hidden_dim=args['node_hidden_dim'],
                            edge_hidden_dim=args['edge_hidden_dim'],
                            num_step_message_passing=args['num_step_message_passing'],
                            num_step_set2set=args['num_step_set2set'],
                            num_layer_set2set=args['num_layer_set2set'],
                            n_tasks=len(args['tasks']),
                            regressor_hidden_feats=args['regressor_hidden_feats'],
                            dropout=args['dropout'])
    elif args['model'] == 'AttentiveFP':
        if args['mode'] == 'parallel':
            model_class = AttentiveFPRegressor
        else:
            model_class = AttentiveFPRegressorBypass
        model = model_class(in_node_feats=in_node_feats,
                            in_edge_feats=in_edge_feats,
                            gnn_out_feats=args['gnn_out_feats'],
                            num_layers=args['num_gnn_layers'],
                            num_timesteps=args['num_timesteps'],
                            n_tasks=len(args['tasks']),
                            regressor_hidden_feats=args['regressor_hidden_feats'],
                            dropout=args['dropout'])

    return model
