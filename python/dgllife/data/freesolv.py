# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# FreeSolv from MoleculeNet for the prediction of hydration free
# energy of small molecules in water

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

__all__ = ['FreeSolv']

class FreeSolv(MoleculeCSVDataset):
    r"""FreeSolv from MoleculeNet for the prediction of hydration free
    energy of small molecules in water

    The dataset provides experimental and calculated hydration free energy
    of small molecules in the water. The calculated values are derived from
    alchemical free energy calculations using molecular dynamics simulations.
    The experimental values are used for labels and the calculated values can
    be used for comparison.

    References:

        * [1] MoleculeNet: A Benchmark for Molecular Machine Learning.
        * [2] FreeSolv: a database of experimental and calculated hydration
              free energies, with input files.

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
        Path to the cached DGLGraphs, default to 'freesolv_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> from dgllife.data import FreeSolv
    >>> from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    >>> dataset = FreeSolv(smiles_to_bigraph, CanonicalAtomFeaturizer())
    >>> # Get size of the dataset
    >>> len(dataset)
    642
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph and hydration free energy
    >>> dataset[0]
    ('CN(C)C(=O)c1ccc(cc1)OC',
     DGLGraph(num_nodes=13, num_edges=26,
              ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
              edata_schemes={}),
     tensor([-11.0100]))

    We also provide information for the iupac name and calculated hydration free energy
    of the compound.

    >>> # Access the information mentioned above for the ith datapoint
    >>> dataset.iupac_names[i]
    >>> dataset.calc_energy[i]

    We can also get all these information along with SMILES, DGLGraph and hydration free
    energy at once.

    >>> dataset.load_full = True
    >>> dataset[0]
    ('CN(C)C(=O)c1ccc(cc1)OC',
     DGLGraph(num_nodes=13, num_edges=26,
              ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
              edata_schemes={}), tensor([-11.0100]),
     '4-methoxy-N,N-dimethyl-benzamide',
     -9.625)
    """
    def __init__(self,
                 smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 log_every=1000,
                 cache_file_path='./freesolv_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/FreeSolv.zip'
        data_path = get_download_dir() + '/FreeSolv.zip'
        dir_path = get_download_dir() + '/FreeSolv'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/SAMPL.csv')

        super(FreeSolv, self).__init__(df=df,
                                       smiles_to_graph=smiles_to_graph,
                                       node_featurizer=node_featurizer,
                                       edge_featurizer=edge_featurizer,
                                       smiles_column='smiles',
                                       cache_file_path=cache_file_path,
                                       task_names=['expt'],
                                       load=load,
                                       log_every=log_every,
                                       init_mask=False,
                                       n_jobs=n_jobs)

        self.load_full = False

        # Iupac names
        self.iupac_names = df['iupac'].tolist()
        self.iupac_names = [self.iupac_names[i] for i in self.valid_ids]
        # Calculated hydration free energy
        self.calc_energy = df['calc'].tolist()
        self.calc_energy = [self.calc_energy[i] for i in self.valid_ids]

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
            IUPAC nomenclature for the ith datapoint, returned only when
            ``self.load_full`` is True.
        float, optional
            Calculated hydration free energy for the ith datapoint, returned only when
            ``self.load_full`` is True.
        """
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.iupac_names[item], self.calc_energy[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]
