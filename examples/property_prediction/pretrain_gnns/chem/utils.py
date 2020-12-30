import torch


def find_neighbor_edges(g, node_id):
    """Given a node with its graph, return all the edges connected to that node."""
    predecessors = g.predecessors(node_id)
    successors = g.successors(node_id)
    predecessors_edges = g.edge_ids(predecessors, torch.full(predecessors.shape, node_id, dtype=torch.long))
    successors_edges = g.edge_ids(torch.full(successors.shape, node_id, dtype=torch.long), successors)
    return torch.cat((predecessors_edges, successors_edges))


def mask_edges(g, masked_nodes_indices):
    """Given a graph and masked nodes, mask all edges that connected to the masked nodes and return these edge indices."""
    masked_edges_indices = []
    for masked_nodes_index in masked_nodes_indices:
        masked_edges_indices.extend(find_neighbor_edges(g, masked_nodes_index))
    return torch.LongTensor(masked_edges_indices)


class PretrainMaskingMoleculeCSVDataset(object):
    """
    adapted from https://lifesci.dgl.ai/_modules/dgllife/data/csv_dataset.html#MoleculeCSVDataset
    Used for pretrain_masking task.
    """

    def __init__(self, df, smiles_to_graph, node_featurizer, edge_featurizer,
                 smiles_column):
        self.df = df
        self.smiles = self.df[smiles_column].tolist()
        self.smiles_to_graph = smiles_to_graph
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer

    def __getitem__(self, item):
        s = self.smiles[item]
        return self.smiles_to_graph(s,
                                    node_featurizer=self.node_featurizer,
                                    edge_featurizer=self.edge_featurizer)

    def __len__(self):
        return len(self.smiles)


class PretrainSupervisedMoleculeCSVDataset(object):
    """
    adapted from https://lifesci.dgl.ai/_modules/dgllife/data/csv_dataset.html#MoleculeCSVDataset
    Used for pretrain_supervised task.
    """

    def __init__(self, data, smiles_to_graph, node_featurizer, edge_featurizer):
        self.data = data
        self.smiles_to_graph = smiles_to_graph
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self._pre_process()

    def __getitem__(self, item):
        s = self.smiles[item]
        graph = self.smiles_to_graph(s,
                                     node_featurizer=self.node_featurizer,
                                     edge_featurizer=self.edge_featurizer)
        label = self.labels[item]
        return graph, label

    def _pre_process(self):
        self.smiles, self.labels = zip(*self.data)

    def __len__(self):
        return len(self.smiles)


class PretrainDataset(object):
    """
    adapted from https://lifesci.dgl.ai/_modules/dgllife/data/csv_dataset.html#MoleculeCSVDataset
    used for pretrain_masking(task=masking) and pretrain_supervised(task=supervised) task.
    """

    def __init__(self, data, smiles_to_graph, node_featurizer, edge_featurizer, task=None):
        self.data = data
        self.smiles_to_graph = smiles_to_graph
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self.task = task
        self._pre_process()

    def __getitem__(self, item):
        s = self.smiles[item]
        graph = self.smiles_to_graph(s,
                                     node_featurizer=self.node_featurizer,
                                     edge_featurizer=self.edge_featurizer)
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
