# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The Toxicology in the 21st Century initiative."""

import dgl.backend as F
import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url

from .csv_dataset import MoleculeCSVDataset
from ..utils.mol_to_graph import smiles_to_bigraph

__all__ = ['Tox21']

class Tox21(MoleculeCSVDataset):
    """Tox21 dataset.

    The Toxicology in the 21st Century (https://tripod.nih.gov/tox21/challenge/)
    initiative created a public database measuring toxicity of compounds, which
    has been used in the 2014 Tox21 Data Challenge. The dataset contains qualitative
    toxicity measurements for 8014 compounds on 12 different targets, including nuclear
    receptors and stress response pathways. Each target results in a binary label.

    A common issue for multi-task prediction is that some datapoints are not labeled for
    all tasks. This is also the case for Tox21. In data pre-processing, we set non-existing
    labels to be 0 so that they can be placed in tensors and used for masking in loss computation.

    All molecules are converted into DGLGraphs. After the first-time construction,
    the DGLGraphs will be saved for reloading so that we do not need to reconstruct them everytime.

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
        featurization methods and need to preprocess from scratch. Default to True.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. Default to 1000.
    cache_file_path : str
        Path to the cached DGLGraphs, default to 'tox21_dglgraph.bin'.
    """
    def __init__(self, smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=True,
                 log_every=1000,
                 cache_file_path='tox21_dglgraph.bin'):
        self._url = 'dataset/tox21.csv.gz'
        data_path = get_download_dir() + '/tox21.csv.gz'
        download(_get_dgl_url(self._url), path=data_path)
        df = pd.read_csv(data_path)
        self.id = df['mol_id']

        df = df.drop(columns=['mol_id'])

        super(Tox21, self).__init__(df, smiles_to_graph, node_featurizer, edge_featurizer,
                                    "smiles", cache_file_path,
                                    load=load, log_every=log_every)
        self._weight_balancing()

    def _weight_balancing(self):
        """Perform re-balancing for each task.

        It's quite common that the number of positive samples and the
        number of negative samples are significantly different. To compensate
        for the class imbalance issue, we can weight each datapoint in
        loss computation.

        In particular, for each task we will set the weight of negative samples
        to be 1 and the weight of positive samples to be the number of negative
        samples divided by the number of positive samples.

        If weight balancing is performed, one attribute will be affected:

        * self._task_pos_weights is set, which is a list of positive sample weights
          for each task.
        """
        num_pos = F.sum(self.labels, dim=0)
        num_indices = F.sum(self.mask, dim=0)
        self._task_pos_weights = (num_indices - num_pos) / num_pos

    @property
    def task_pos_weights(self):
        """Get weights for positive samples on each task

        It's quite common that the number of positive samples and the
        number of negative samples are significantly different. To compensate
        for the class imbalance issue, we can weight each datapoint in
        loss computation.

        In particular, for each task we will set the weight of negative samples
        to be 1 and the weight of positive samples to be the number of negative
        samples divided by the number of positive samples.

        Returns
        -------
        Tensor of dtype float32 and shape (T)
            Weight of positive samples on all tasks
        """
        return self._task_pos_weights
