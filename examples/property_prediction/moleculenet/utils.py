# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import numpy as np
import random
import torch

from dgllife.utils.splitters import RandomSplitter

def set_random_seed(seed=0):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_dataset_for_classification(args):
    """Load dataset for classification tasks.

    Parameters
    ----------
    args : dict
        Configurations.

    Returns
    -------
    dataset
        The whole dataset.
    train_set
        Subset for training.
    val_set
        Subset for validation.
    test_set
        Subset for test.
    """
    assert args['dataset'] in ['Tox21']
    if args['dataset'] == 'Tox21':
        from dgllife.data import Tox21
        dataset = Tox21(smiles_to_graph=args['smiles_to_graph'],
                        node_featurizer=args.get('node_featurizer', None),
                        edge_featurizer=args.get('edge_featurizer', None),
                        load=False,
                        cache_file_path=args['exp'])
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'], random_state=args['random_seed'])

    return dataset, train_set, val_set, test_set

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.

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
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
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
                             hidden_feats=args['gcn_hidden_feats'],
                             classifier_hidden_feats=args['classifier_hidden_feats'],
                             n_tasks=args['n_tasks'])

    if args['model'] == 'GAT':
        from dgllife.model import GATPredictor
        model = GATPredictor(in_feats=args['node_featurizer'].feat_size(),
                             hidden_feats=args['gat_hidden_feats'],
                             num_heads=args['num_heads'],
                             classifier_hidden_feats=args['classifier_hidden_feats'],
                             n_tasks=args['n_tasks'])

    if args['model'] == 'Weave':
        from dgllife.model import WeavePredictor
        model = WeavePredictor(node_in_feats=args['node_featurizer'].feat_size(),
                               edge_in_feats=args['edge_featurizer'].feat_size(),
                               num_gnn_layers=args['num_gnn_layers'],
                               gnn_hidden_feats=args['gnn_hidden_feats'],
                               graph_feats=args['graph_feats'],
                               n_tasks=args['n_tasks'])

    return model
