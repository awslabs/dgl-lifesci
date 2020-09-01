# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ClinTox from MoleculeNet for the prediction of clinical trial toxicity
# (or absence of toxicity) and FDA approval status

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

__all__ = ['ClinTox']

class ClinTox(MoleculeCSVDataset):
    r"""ClinTox from MoleculeNet for the prediction of clinical trial toxicity
    (or absence of toxicity) and FDA approval status

    The ClinTox dataset compares drugs approved by the FDA and drugs that have failed
    clinical trials for toxicity reasons. The dataset includes two classification
    tasks for 1491 drug compounds with known chemical structures: (1) clinical trial
    toxicity (or absence of toxicity) and (2) FDA approval status. The MoleculeNet
    benchmark compiles the list of FDA-approved drugs from the SWEETLEAD database and
    the list of drugs that failed clinical trials for toxicity reasons from the
    Aggregate Analysis of ClinicalTrials.gov (AACT) database.

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
        Path to the cached DGLGraphs, default to 'clintox_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> import torch
    >>> from dgllife.data import ClinTox
    >>> from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    >>> dataset = ClinTox(smiles_to_bigraph, CanonicalAtomFeaturizer())
    >>> # Get size of the dataset
    >>> len(dataset)
    1478
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph, labels, and masks
    >>> dataset[0]
    ('*C(=O)[C@H](CCCCNC(=O)OCCOC)NC(=O)OCCOC',
     Graph(num_nodes=24, num_edges=46,
           ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
           edata_schemes={}),
     tensor([1., 0.]),
     tensor([1., 1.]))

    To address the imbalance between positive and negative samples, we can re-weight
    positive samples for each task based on the training datapoints.

    >>> train_ids = torch.arange(500)
    >>> dataset.task_pos_weights(train_ids)
    tensor([ 0.0684, 10.9048])
    """
    def __init__(self,
                 smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 log_every=1000,
                 cache_file_path='./clintox_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/clintox.zip'
        data_path = get_download_dir() + '/clintox.zip'
        dir_path = get_download_dir() + '/clintox'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/clintox.csv')

        super(ClinTox, self).__init__(df=df,
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
