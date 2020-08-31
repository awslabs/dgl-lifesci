# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# BACE from MoleculeNet for the prediction of quantitative (IC50) and
# qualitative (binary label) binding results for a set of inhibitors of
# human beta-secretase 1 (BACE-1).

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

__all__ = ['BACE']

class BACE(MoleculeCSVDataset):
    r"""BACE from MoleculeNet for the prediction of quantitative and qualitative binding results
    for a set of inhibitors of human beta-secretase 1 (BACE-1)

    The dataset contains experimental values reported in scientific literature over the past
    decade, some with detailed crystal structures available. The MoleculeNet benchmark merged
    a collection of 1522 compounds with their 2D structures and binary labels.

    References:

        * [1] MoleculeNet: A Benchmark for Molecular Machine Learning.

    Parameters
    ----------
    smiles_to_graph: callable, str -> DGLGraph
        A function turning a SMILES string into a DGLGraph.
        Default to :func:`dgllife.utils.smiles_to_bigraph`.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to False.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. Default to 1000.
    cache_file_path : str
        Path to the cached DGLGraphs, default to 'bace_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> import torch
    >>> from dgllife.data import BACE
    >>> from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    >>> dataset = BACE(smiles_to_bigraph, CanonicalAtomFeaturizer())
    >>> # Get size of the dataset
    >>> len(dataset)
    1513
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph, labels, and masks
    >>> dataset[0]
    ('O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C',
     Graph(num_nodes=32, num_edges=70,
           ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
           edata_schemes={}),
     tensor([1.]),
     tensor([1.]))

    The dataset instance also contains information about molecule ids.

    >>> dataset.ids[i]

    We can also get the id along with SMILES, DGLGraph, labels, and masks at once.

    >>> dataset.load_full = True
    >>> dataset[0]
    ('O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C',
     Graph(num_nodes=32, num_edges=70,
           ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
           edata_schemes={}),
     tensor([1.]),
     tensor([1.]),
     'BACE_1')

    To address the imbalance between positive and negative samples, we can re-weight
    positive samples for each task based on the training datapoints.

    >>> train_ids = torch.arange(500)
    >>> dataset.task_pos_weights(train_ids)
    tensor([0.2594])
    """
    def __init__(self,
                 smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 log_every=1000,
                 cache_file_path='./bace_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/bace.zip'
        data_path = get_download_dir() + '/bace.zip'
        dir_path = get_download_dir() + '/bace'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/bace.csv')

        super(BACE, self).__init__(df=df,
                                   smiles_to_graph=smiles_to_graph,
                                   node_featurizer=node_featurizer,
                                   edge_featurizer=edge_featurizer,
                                   smiles_column='mol',
                                   cache_file_path=cache_file_path,
                                   task_names=['Class'],
                                   load=load,
                                   log_every=log_every,
                                   init_mask=True,
                                   n_jobs=n_jobs)

        self.load_full = False
        self.ids = df['CID'].tolist()
        self.ids = [self.ids[i] for i in self.valid_ids]

    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32 and shape (T)
            Labels of the ith datapoint for all tasks. T for the number of tasks.
        Tensor of dtype float32 and shape (T)
            Binary masks of the ith datapoint indicating the existence of labels for all tasks.
        str, optional
            Id for the ith datapoint, returned only when ``self.load_full`` is True.
        """
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.ids[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
