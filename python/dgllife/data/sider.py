# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sider from MoleculeNet for the prediction of drug side-effects

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

__all__ = ['SIDER']

class SIDER(MoleculeCSVDataset):
    r"""SIDER from MoleculeNet for the prediction of grouped drug side-effects

    The Side Effect Resource (SIDER) is a database of marketed drugs and adverse drug relations
    (ADR). The MoleculeNet benchmark has grouped drug side-effects into 27 system organ classes
    following MedDRA classifications measured for 1427 approved drugs.

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
        Path to the cached DGLGraphs, default to 'sider_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> import torch
    >>> from dgllife.data import SIDER
    >>> from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    >>> dataset = SIDER(smiles_to_bigraph, CanonicalAtomFeaturizer())
    >>> # Get size of the dataset
    >>> len(dataset)
    1427
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph, labels, and masks
    >>> dataset[0]
    ('C(CNCCNCCNCCN)N',
     Graph(num_nodes=13, num_edges=24,
           ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
           edata_schemes={}),
     tensor([1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0.,
             0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0.]),
     tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))

    To address the imbalance between positive and negative samples, we can re-weight
    positive samples for each task based on the training datapoints.

    >>> train_ids = torch.arange(500)
    >>> dataset.task_pos_weights(train_ids)
    tensor([ 1.1368,  0.4793, 49.0000,  0.7123,  0.2626,  0.5015,  0.1211,  5.2500,
             0.4205,  1.0325,  3.1667,  0.1312,  3.9505,  5.9444,  0.3263,  0.7544,
             0.0823,  4.9524,  0.3889,  0.3812,  0.4706,  0.6447, 11.5000,  1.4272,
             0.5060,  0.1136,  0.5106])
    """
    def __init__(self,
                 smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 log_every=1000,
                 cache_file_path='./sider_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/sider.zip'
        data_path = get_download_dir() + '/sider.zip'
        dir_path = get_download_dir() + '/sider'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/sider.csv')

        super(SIDER, self).__init__(df=df,
                                    smiles_to_graph=smiles_to_graph,
                                    node_featurizer=node_featurizer,
                                    edge_featurizer=edge_featurizer,
                                    smiles_column='smiles',
                                    cache_file_path=cache_file_path,
                                    load=load,
                                    log_every=log_every,
                                    init_mask=True,
                                    n_jobs=n_jobs)

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
        """
        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
