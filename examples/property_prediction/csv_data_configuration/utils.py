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
    with open('model_configures/{}.json'.format(model), 'w') as f:
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

def load_model(args):
    if args['model'] == 'GCN':
        from dgllife.model import GCNPredictor
        model = GCNPredictor(in_feats=args['node_featurizer'].feat_size(),
                             hidden_feats=[args['gnn_hidden_feats']] * args['num_gnn_layers'],
                             activation=[F.relu] * args['num_gnn_layers'],
                             residual=[args['residual']] * args['num_gnn_layers'],
                             batchnorm=[args['batchnorm']] * args['num_gnn_layers'],
                             dropout=[args['dropout']] * args['num_gnn_layers'],
                             predictor_hidden_feats=args['predictor_hidden_feats'],
                             predictor_dropout=args['dropout'],
                             n_tasks=args['n_tasks'])
    else:
        return ValueError("Expect model to be 'GCN', got {}".format(args['model']))

    return model
