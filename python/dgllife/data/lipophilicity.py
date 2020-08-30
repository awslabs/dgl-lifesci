# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lipophilicity from MoleculeNet for the prediction of octanol/water
# distribution coefficient (logD at pH 7.4) of 4200 compounds

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

__all__ = ['Lipophilicity']

class Lipophilicity(MoleculeCSVDataset):
    r"""Lipophilicity from MoleculeNet for the prediction of octanol/water
    distribution coefficient (logD at pH 7.4) of 4200 compounds

    This dataset is curated from ChEMBL database containing experimental results
    on octanol/water distribution coefficient (logD at pH=7.4). Due to the importance
    of lipophilicity in membrane permeability and solubility, the task is of high
    importance to drug development.

    References:

        * [1] MoleculeNet: A Benchmark for Molecular Machine Learning.
        * [2] ChEMBL Deposited Data Set - AZ dataset; 2015.

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
        Path to the cached DGLGraphs, default to 'lipophilicity_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> from dgllife.data import Lipophilicity
    >>> from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    >>> dataset = Lipophilicity(smiles_to_bigraph, CanonicalAtomFeaturizer())
    >>> # Get size of the dataset
    >>> len(dataset)
    4200
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph and logD
    >>> dataset[0]
    ('Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14',
     DGLGraph(num_nodes=24, num_edges=54,
              ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
              edata_schemes={}),
     tensor([3.5400]))

    We also provide information for the ChEMBL id of the compound.

    >>> dataset.chembl_ids[i]

    We can also get the ChEMBL id along with SMILES, DGLGraph and logD at once.

    >>> dataset.load_full = True
    >>> dataset[0]
    ('Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14',
     DGLGraph(num_nodes=24, num_edges=54,
              ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
              edata_schemes={}),
     tensor([3.5400]),
     'CHEMBL596271')
    """
    def __init__(self,
                 smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 log_every=1000,
                 cache_file_path='./lipophilicity_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/lipophilicity.zip'
        data_path = get_download_dir() + '/lipophilicity.zip'
        dir_path = get_download_dir() + '/lipophilicity'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/Lipophilicity.csv')

        super(Lipophilicity, self).__init__(df=df,
                                            smiles_to_graph=smiles_to_graph,
                                            node_featurizer=node_featurizer,
                                            edge_featurizer=edge_featurizer,
                                            smiles_column='smiles',
                                            cache_file_path=cache_file_path,
                                            task_names=['exp'],
                                            load=load,
                                            log_every=log_every,
                                            init_mask=False,
                                            n_jobs=n_jobs)

        self.load_full = False

        # ChEMBL ids
        self.chembl_ids = df['CMPD_CHEMBLID'].tolist()
        self.chembl_ids = [self.chembl_ids[i] for i in self.valid_ids]

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
        Tensor of dtype float32 and shape (1)
            Labels of the ith datapoint
        str, optional
            ChEMBL id of the ith datapoint, returned only when
            ``self.load_full`` is True.
        """
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], self.chembl_ids[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]
