# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import errno
import json
import os
import torch
import torch.nn.functional as F

from dgllife.utils import ScaffoldSplitter

def init_featurizer(args):
    """Initialize node/edge featurizer

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with featurizers updated
    """
    if args['atom_featurizer_type'] == 'canonical':
        from dgllife.utils import CanonicalAtomFeaturizer
        args['node_featurizer'] = CanonicalAtomFeaturizer()
    elif args['atom_featurizer_type'] == 'attentivefp':
        from dgllife.utils import AttentiveFPAtomFeaturizer
        args['node_featurizer'] = AttentiveFPAtomFeaturizer()
    else:
        return ValueError(
            "Expect node_featurizer to be in ['canonical', 'attentivefp'], "
            "got {}".format(args['atom_featurizer_type']))

    if args['model'] in ['Weave', 'MPNN', 'AttentiveFP']:
        if args['bond_featurizer_type'] == 'canonical':
            from dgllife.utils import CanonicalBondFeaturizer
            args['edge_featurizer'] = CanonicalBondFeaturizer()
        elif args['bond_featurizer_type'] == 'attentivefp':
            from dgllife.utils import AttentiveFPBondFeaturizer
            args['edge_featurizer'] = AttentiveFPBondFeaturizer()
    else:
        args['edge_featurizer'] = None

    return args

def get_configure(model):
    """Query for the manually specified configuration

    Parameters
    ----------
    model : str
        Model type

    Returns
    -------
    dict
        Returns the manually specified configuration
    """
    with open('model_configures/{}.json'.format(model), 'r') as f:
        config = json.load(f)
    return config

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

def init_trial_path(args):
    """Initialize the path for a hyperparameter setting

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with the trial path updated
    """
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = args['result_path'] + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    args['trial_path'] = path_to_results
    mkdir_p(args['trial_path'])

    return args

def split_dataset(args, dataset):
    """Split the dataset

    Parameters
    ----------
    args : dict
        Settings
    dataset
        Dataset instance

    Returns
    -------
    train_set
        Training subset
    val_set
        Validation subset
    test_set
        Test subset
    """
    train_ratio, val_ratio, test_ratio = map(float, args['split_ratio'].split(','))
    if args['split'] == 'scaffold':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    else:
        return ValueError("Expect the splitting method to be 'scaffold', got {}".format(args['split']))

    return train_set, val_set, test_set

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

def collate_molgraphs_unlabeled(data):
    """Batching a list of datapoints without labels

    Parameters
    ----------
    data : list of 2-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES and a DGLGraph.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    """
    smiles, graphs = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    return smiles, bg

def load_model(exp_configure):
    if exp_configure['model'] == 'GCN':
        from dgllife.model import GCNPredictor
        model = GCNPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'])
    elif exp_configure['model'] == 'GAT':
        from dgllife.model import GATPredictor
        model = GATPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            num_heads=[exp_configure['num_heads']] * exp_configure['num_gnn_layers'],
            feat_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            attn_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            alphas=[exp_configure['alpha']] * exp_configure['num_gnn_layers'],
            residuals=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'Weave':
        from dgllife.model import WeavePredictor
        model = WeavePredictor(
            node_in_feats=exp_configure['in_node_feats'],
            edge_in_feats=exp_configure['in_edge_feats'],
            num_gnn_layers=exp_configure['num_gnn_layers'],
            gnn_hidden_feats=exp_configure['gnn_hidden_feats'],
            graph_feats=exp_configure['graph_feats'],
            gaussian_expand=exp_configure['gaussian_expand'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'MPNN':
        from dgllife.model import MPNNPredictor
        model = MPNNPredictor(
            node_in_feats=exp_configure['in_node_feats'],
            edge_in_feats=exp_configure['in_edge_feats'],
            node_out_feats=exp_configure['node_out_feats'],
            edge_hidden_feats=exp_configure['edge_hidden_feats'],
            num_step_message_passing=exp_configure['num_step_message_passing'],
            num_step_set2set=exp_configure['num_step_set2set'],
            num_layer_set2set=exp_configure['num_layer_set2set'],
            n_tasks=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'AttentiveFP':
        from dgllife.model import AttentiveFPPredictor
        model = AttentiveFPPredictor(
            node_feat_size=exp_configure['in_node_feats'],
            edge_feat_size=exp_configure['in_edge_feats'],
            num_layers=exp_configure['num_layers'],
            num_timesteps=exp_configure['num_timesteps'],
            graph_feat_size=exp_configure['graph_feat_size'],
            dropout=exp_configure['dropout']
        )
    else:
        return ValueError("Expect model to be from ['GCN', 'GAT'], "
                          "got {}".format(exp_configure['model']))

    return model

def predict(args, model, bg):
    node_feats = bg.ndata.pop('h').to(args['device'])
    if args['edge_featurizer'] is None:
        return model(bg, node_feats)
    else:
        edge_feats = bg.edata.pop('e').to(args['device'])
        return model(bg, node_feats, edge_feats)
