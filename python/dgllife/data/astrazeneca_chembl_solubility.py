# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Experimental solubility determined at AstraZeneca on a
# set of compounds, recorded in ChEMBL

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

__all__ = ['AstraZenecaChEMBLSolubility']

class AstraZenecaChEMBLSolubility(MoleculeCSVDataset):
    r"""Experimental solubility determined at AstraZeneca, extracted from ChEMBL

    The dataset provides experimental solubility (in nM unit) for 1763 molecules
    in pH7.4 using solid starting material using the method described in [1].

    References:

        * [1] A Highly Automated Assay for Determining the Aqueous Equilibrium
          Solubility of Drug Discovery Compounds
        * [2] `CHEMBL3301361 <https://www.ebi.ac.uk/chembl/document_report_card/CHEMBL3301361/>`__

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
        Path to the cached DGLGraphs. Default to 'AstraZeneca_chembl_solubility_graph.bin'.
    log_of_values : bool
        Whether to take the logarithm of the solubility values. Before taking the logarithm,
        the values can have a range of [100, 1513600]. After taking the logarithm, the
        values will have a range of [4.61, 14.23]. Default to True.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> from dgllife.data import AstraZenecaChEMBLSolubility
    >>> from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    >>> dataset = AstraZenecaChEMBLSolubility(smiles_to_bigraph, CanonicalAtomFeaturizer())
    >>> # Get size of the dataset
    >>> len(dataset)
    1763
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph and solubility
    >>> dataset[0]
    ('Cc1nc(C)c(-c2ccc([C@H]3CC[C@H](Cc4nnn[nH]4)CC3)cc2)nc1C(N)=O',
     DGLGraph(num_nodes=29, num_edges=64,
              ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
              edata_schemes={}),
     tensor([12.5032]))

    We also provide information for the ChEMBL id and molecular weight of the compound.

    >>> dataset.chembl_ids[i]
    >>> dataset.mol_weight[i]

    We can also get the ChEMBL id and molecular weight along with SMILES, DGLGraph and
    solubility at once.

    >>> dataset.load_full = True
    >>> dataset[0]
    ('Cc1nc(C)c(-c2ccc([C@H]3CC[C@H](Cc4nnn[nH]4)CC3)cc2)nc1C(N)=O',
     DGLGraph(num_nodes=29, num_edges=64,
              ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
              edata_schemes={}),
     tensor([12.5032]),
     'CHEMBL2178940',
     391.48)
    """
    def __init__(self,
                 smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 log_every=1000,
                 cache_file_path='./AstraZeneca_chembl_solubility_graph.bin',
                 log_of_values=True,
                 n_jobs=1):

        self._url = 'dataset/AstraZeneca_ChEMBL_Solubility.csv'
        data_path = get_download_dir() + '/AstraZeneca_ChEMBL_Solubility.csv'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        df = pd.read_csv(data_path)

        super(AstraZenecaChEMBLSolubility, self).__init__(
            df=df,
            smiles_to_graph=smiles_to_graph,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            smiles_column='Smiles',
            cache_file_path=cache_file_path,
            task_names=['Solubility'],
            load=load,
            log_every=log_every,
            init_mask=False,
            n_jobs=n_jobs)

        self.load_full = False
        # ChEMBL ids
        self.chembl_ids = df['Molecule ChEMBL ID'].tolist()
        self.chembl_ids = [self.chembl_ids[i] for i in self.valid_ids]
        # Molecular weight
        self.mol_weight = df['Molecular Weight'].tolist()
        self.mol_weight = [self.mol_weight[i] for i in self.valid_ids]

        if log_of_values:
            self.labels = self.labels.log()

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
            ChEMBL id of the ith datapoint, returned only when ``self.load_full`` is True.
        float, optional
            Molecular weight of the ith datapoint, returned only when ``self.load_full`` is True.
        """
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.chembl_ids[item], self.mol_weight[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]
