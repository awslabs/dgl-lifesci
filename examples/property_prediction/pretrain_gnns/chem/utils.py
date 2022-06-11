# -*- coding: utf-8 -*-
import torch
import dgl
from dgllife.utils import ScaffoldSplitter, RandomSplitter


def find_neighbor_edges(g, node_id):
    """Given a node with its graph, return all the edges connected to that node."""
    predecessors = g.predecessors(node_id)
    successors = g.successors(node_id)
    predecessors_edges = g.edge_ids(predecessors, torch.full(predecessors.shape, node_id, dtype=torch.int))
    successors_edges = g.edge_ids(torch.full(successors.shape, node_id, dtype=torch.int), successors)
    return torch.cat((predecessors_edges, successors_edges))


def mask_edges(g, masked_nodes_indices):
    """Given a graph and masked nodes, mask all edges that connected to the masked nodes and return these edge indices."""
    masked_edges_indices = []
    for masked_nodes_index in masked_nodes_indices:
        masked_edges_indices.extend(find_neighbor_edges(g, masked_nodes_index.int()))
    return torch.LongTensor(masked_edges_indices)


class PretrainDataset(object):
    """
    adapted from https://lifesci.dgl.ai/_modules/dgllife/data/csv_dataset.html#MoleculeCSVDataset
    used for pretrain_masking(task=masking) and pretrain_supervised(task=supervised) task.
    """

    def __init__(self, data, smiles_to_graph, smiles_column=None, task=None):
        self.data = data
        self.smiles_column = smiles_column
        if task == 'masking':
            self.smiles = self.data[smiles_column].tolist()
        self.smiles_to_graph = smiles_to_graph
        self.task = task
        self._pre_process()

    def __getitem__(self, item):
        s = self.smiles[item]
        graph = self.smiles_to_graph(s)
        if self.task == 'masking':
            return graph
        elif self.task == 'supervised':
            label = self.labels[item]
            return graph, label
        else:
            raise ValueError('Dataset task should be either `masking` or `supervised`.')

    def _pre_process(self):
        if self.task == 'supervised':
            self.smiles, self.labels = zip(*self.data)
        elif self.task == 'masking':
            pass
        else:
            raise ValueError('Dataset task should be either `masking` or `supervised`.')

    def __len__(self):
        return len(self.smiles)


def split_dataset(args, dataset):
    """Split the dataset for pretrain downstream task
    Parameters
    ----------
    args
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
    train_ratio, val_ratio, test_ratio = map(float, args.split_ratio.split(','))
    if args.split == 'scaffold':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')
    elif args.split == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    else:
        return ValueError("Expect the splitting method to be 'scaffold' or 'random', got {}".format(args.split))

    return train_set, val_set, test_set


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
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
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks
