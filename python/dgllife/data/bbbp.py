# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# BBBP from MoleculeNet for the prediction of permeability properties.

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

__all__ = ['BBBP']

class BBBP(MoleculeCSVDataset):
    r"""BBBP from MoleculeNet for the prediction of permeability properties

    The Blood-brain barrier penetration (BBBP) dataset comes from a study on
    the modeling and prediction of the barrier permeability. As a membrane
    separating circulating blood and brain extracellular fluid, the blood-brain
    barrier blocks most drugs, hormones and neurotransmitters. Thus penetration of
    the barrier forms a long-standing issue in development of drugs targeting
    central nervous system. This dataset includes binary labels for over 2000
    compounds on their permeability properties.

    References:

        * [1] MoleculeNet: A Benchmark for Molecular Machine Learning.
        * [2] A Bayesian approach to in silico blood-brain barrier penetration modeling

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
        Path to the cached DGLGraphs, default to 'bbbp_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> import torch
    >>> from dgllife.data import BBBP
    >>> from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    >>> dataset = BBBP(smiles_to_bigraph, CanonicalAtomFeaturizer())
    >>> # Get size of the dataset
    >>> len(dataset)
    2039
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph, labels, and masks
    >>> dataset[0]
    ('[Cl].CC(C)NCC(O)COc1cccc2ccccc12',
     Graph(num_nodes=20, num_edges=40,
           ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
           edata_schemes={}),
     tensor([1.]),
     tensor([1.]))

    The dataset instance also contains information about compound name.

    >>> dataset.names[i]

    We can also get the name along with SMILES, DGLGraph, labels, and masks at once.

    >>> dataset.load_full = True
    >>> dataset[0]
    ('[Cl].CC(C)NCC(O)COc1cccc2ccccc12',
     Graph(num_nodes=20, num_edges=40,
           ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
           edata_schemes={}),
     tensor([1.]),
     tensor([1.]),
     'Propanolol')

    To address the imbalance between positive and negative samples, we can re-weight
    positive samples for each task based on the training datapoints.

    >>> train_ids = torch.arange(500)
    >>> dataset.task_pos_weights(train_ids)
    tensor([0.7123])
    """
    def __init__(self,
                 smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 log_every=1000,
                 cache_file_path='./bbbp_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/bbbp.zip'
        data_path = get_download_dir() + '/bbbp.zip'
        dir_path = get_download_dir() + '/bbbp'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/BBBP.csv')

        super(BBBP, self).__init__(df=df,
                                   smiles_to_graph=smiles_to_graph,
                                   node_featurizer=node_featurizer,
                                   edge_featurizer=edge_featurizer,
                                   smiles_column='smiles',
                                   cache_file_path=cache_file_path,
                                   task_names=['p_np'],
                                   load=load,
                                   log_every=log_every,
                                   init_mask=True,
                                   n_jobs=n_jobs)

        self.load_full = False
        self.names = df['name'].tolist()
        self.names = [self.names[i] for i in self.valid_ids]

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
            Name for the ith compound, returned only when ``self.load_full`` is True.
        """
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.names[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
