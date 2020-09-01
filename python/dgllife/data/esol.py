# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ESOL from MoleculeNet for the prediction of water solubility

import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

__all__ = ['ESOL']

class ESOL(MoleculeCSVDataset):
    r"""ESOL from MoleculeNet for the prediction of water solubility

    ESOL (delaney) is a standard regression dataset containing structures and water solubility
    data for 1128 compounds. It uses measured log solubility in mols per litre of compounds
    as labels. The dataset is widely used to validate machine learning models on estimating
    solubility directly from molecular structures. Note that these structures don't include
    3D coordinates, since solubility is a property of a molecule and not of its particular
    conformers.

    References:

        * [1] MoleculeNet: A Benchmark for Molecular Machine Learning.
        * [2] ESOL: estimating aqueous solubility directly from molecular structure.

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
        Path to the cached DGLGraphs, default to 'esol_dglgraph.bin'.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.

    Examples
    --------

    >>> from dgllife.data import ESOL
    >>> from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    >>> dataset = ESOL(smiles_to_bigraph, CanonicalAtomFeaturizer())
    >>> # Get size of the dataset
    >>> len(dataset)
    1128
    >>> # Get the 0th datapoint, consisting of SMILES, DGLGraph and solubility
    >>> dataset[0]
    ('OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ',
     DGLGraph(num_nodes=32, num_edges=68,
              ndata_schemes={}
              edata_schemes={}),
     tensor([-0.7700]))

    We also provide information for the name, estimated solubility, minimum atom
    degree, molecular weight, number of h bond donors, number of rings,
    number of rotatable bonds, and polar surface area of the compound

    >>> # Access the information mentioned above for the ith datapoint
    >>> dataset.compound_names[i]
    >>> dataset.estimated_solubility[i]
    >>> dataset.min_degree[i]
    >>> dataset.mol_weight[i]
    >>> dataset.num_h_bond_donors[i]
    >>> dataset.num_rings[i]
    >>> dataset.num_rotatable_bonds[i]
    >>> dataset.polar_surface_area[i]

    We can also get all these information along with SMILES, DGLGraph and solubility
    at once.

    >>> dataset.load_full = True
    >>> dataset[0]
    ('OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ',
     DGLGraph(num_nodes=32, num_edges=68,
              ndata_schemes={}
              edata_schemes={}),
     tensor([-0.7700]),
     'Amigdalin',
     -0.974,
     1,
     457.43200000000013,
     7,
     3,
     7,
     202.32)
    """
    def __init__(self,
                 smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 log_every=1000,
                 cache_file_path='./esol_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/ESOL.zip'
        data_path = get_download_dir() + '/ESOL.zip'
        dir_path = get_download_dir() + '/ESOL'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/delaney-processed.csv')

        super(ESOL, self).__init__(df=df,
                                   smiles_to_graph=smiles_to_graph,
                                   node_featurizer=node_featurizer,
                                   edge_featurizer=edge_featurizer,
                                   smiles_column='smiles',
                                   cache_file_path=cache_file_path,
                                   task_names=['measured log solubility in mols per litre'],
                                   load=load,
                                   log_every=log_every,
                                   init_mask=False,
                                   n_jobs=n_jobs)

        self.load_full = False
        # Compound names in PubChem
        self.compound_names = df['Compound ID'].tolist()
        self.compound_names = [self.compound_names[i] for i in self.valid_ids]
        # Estimated solubility
        self.estimated_solubility = df['ESOL predicted log solubility in mols per litre'].tolist()
        self.estimated_solubility = [self.estimated_solubility[i] for i in self.valid_ids]
        # Minimum atom degree
        self.min_degree = df['Minimum Degree'].tolist()
        self.min_degree = [self.min_degree[i] for i in self.valid_ids]
        # Molecular weight
        self.mol_weight = df['Molecular Weight'].tolist()
        self.mol_weight = [self.mol_weight[i] for i in self.valid_ids]
        # Number of H-Bond Donors
        self.num_h_bond_donors = df['Number of H-Bond Donors'].tolist()
        self.num_h_bond_donors = [self.num_h_bond_donors[i] for i in self.valid_ids]
        # Number of rings
        self.num_rings = df['Number of Rings'].tolist()
        self.num_rings = [self.num_rings[i] for i in self.valid_ids]
        # Number of rotatable bonds
        self.num_rotatable_bonds = df['Number of Rotatable Bonds'].tolist()
        self.num_rotatable_bonds = [self.num_rotatable_bonds[i] for i in self.valid_ids]
        # Polar Surface Area
        self.polar_surface_area = df['Polar Surface Area'].tolist()
        self.polar_surface_area = [self.polar_surface_area[i] for i in self.valid_ids]

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
            Name for the ith compound, returned only when ``self.load_full`` is True.
        float, optional
            Estimated solubility for the ith compound,
            returned only when ``self.load_full`` is True.
        int, optional
            Minimum atom degree for the ith datapoint, returned only when
            ``self.load_full`` is True.
        float, optional
            Molecular weight for the ith datapoint, returned only when
            ``self.load_full`` is True.
        int, optional
            Number of h bond donors for the ith datapoint, returned only when
            ``self.load_full`` is True.
        int, optional
            Number of rings in the ith datapoint, returned only when
            ``self.load_full`` is True.
        int, optional
            Number of rotatable bonds in the ith datapoint, returned only when
            ``self.load_full`` is True.
        float, optional
            Polar surface area for the ith datapoint, returned only when
            ``self.load_full`` is True.
        """
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.compound_names[item], self.estimated_solubility[item], \
                   self.min_degree[item], self.mol_weight[item], \
                   self.num_h_bond_donors[item], self.num_rings[item], \
                   self.num_rotatable_bonds[item], self.polar_surface_area[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]
