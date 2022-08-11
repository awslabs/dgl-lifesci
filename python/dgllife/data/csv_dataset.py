# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Creating datasets from .csv files for molecular property prediction.

import dgl.backend as F
import numpy as np
import os
import pandas as pd
import torch

from dgl.data.utils import save_graphs, load_graphs
from functools import partial

from ..utils.io import pmap
from ..utils.mol_to_graph import ToGraph, SMILESToBigraph

__all__ = ['MoleculeCSVDataset']

class MoleculeCSVDataset(object):
    """MoleculeCSVDataset

    This is a general class for loading molecular data from :class:`pandas.DataFrame`.

    In data pre-processing, we construct a binary mask indicating the existence of labels.

    All molecules are converted into DGLGraphs. After the first-time construction, the
    DGLGraphs can be saved for reloading so that we do not need to reconstruct them every time.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe including smiles and labels. Can be loaded by pandas.read_csv(file_path).
        One column includes smiles and some other columns include labels.
    smiles_to_graph: callable, str -> DGLGraph
        A function turning a SMILES string into a DGLGraph. If None, it uses
        :func:`dgllife.utils.SMILESToBigraph` by default.
    node_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    edge_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    smiles_column: str
        Column name for smiles in ``df``.
    cache_file_path: str
        Path to store the preprocessed DGLGraphs. For example, this can be ``'dglgraph.bin'``.
    task_names : list of str or None, optional
        Columns in the data frame corresponding to real-valued labels. If None, we assume
        all columns except the smiles_column are labels. Default to None.
    load : bool, optional
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to False.
    log_every : bool, optional
        Print a message every time ``log_every`` molecules are processed. It only comes
        into effect when :attr:`n_jobs` is greater than 1. Default to 1000.
    init_mask : bool, optional
        Whether to initialize a binary mask indicating the existence of labels. Default to True.
    n_jobs : int, optional
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.
    error_log : str, optional
        Path to a CSV file of molecules that RDKit failed to parse. If not specified,
        the molecules will not be recorded.
    """
    def __init__(self, df, smiles_to_graph=None, node_featurizer=None, edge_featurizer=None,
                 smiles_column=None, cache_file_path=None, task_names=None, load=False,
                 log_every=1000, init_mask=True, n_jobs=1, error_log=None):
        self.df = df
        self.smiles = self.df[smiles_column].tolist()
        if task_names is None:
            self.task_names = self.df.columns.drop([smiles_column]).tolist()
        else:
            self.task_names = task_names
        self.n_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path

        if isinstance(smiles_to_graph, ToGraph):
            assert node_featurizer is None, \
                'Initialize smiles_to_graph object with node_featurizer=node_featurizer'
            assert edge_featurizer is None, \
                'Initialize smiles_to_graph object with edge_featurizer=edge_featurizer'
        elif smiles_to_graph is None:
            smiles_to_graph = SMILESToBigraph(node_featurizer=node_featurizer,
                                              edge_featurizer=edge_featurizer)
        else:
            smiles_to_graph = partial(smiles_to_graph, node_featurizer=node_featurizer,
                                      edge_featurizer=edge_featurizer)

        self._pre_process(smiles_to_graph, load, log_every, init_mask, n_jobs, error_log)

        # Only useful for binary classification tasks
        self._task_pos_weights = None

    def _pre_process(self, smiles_to_graph, load, log_every, init_mask, n_jobs, error_log):
        """Pre-process the dataset

        * Convert molecules from smiles format into DGLGraphs
          and featurize their atoms
        * Set missing labels to be 0 and use a binary masking
          matrix to mask them

        Parameters
        ----------
        smiles_to_graph : callable, SMILES -> DGLGraph
            Function for converting a SMILES (str) into a DGLGraph.
        load : bool
            Whether to load the previously pre-processed dataset or pre-process from scratch.
            ``load`` should be False when we want to try different graph construction and
            featurization methods and need to preprocess from scratch. Default to True.
        log_every : bool
            Print a message every time ``log_every`` molecules are processed. It only comes
            into effect when :attr:`n_jobs` is greater than 1.
        init_mask : bool
            Whether to initialize a binary mask indicating the existence of labels.
        n_jobs : int
            Degree of parallelism for pre processing. Default to 1.
        error_log : str
            Path to a CSV file of molecules that RDKit failed to parse. If not specified,
            the molecules will not be recorded.
        """
        if os.path.exists(self.cache_file_path) and load:
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
            self.labels = label_dict['labels']
            if init_mask:
                self.mask = label_dict['mask']
            else:
                self.mask = None
            self.valid_ids = label_dict['valid_ids'].tolist()
        else:
            print('Processing dgl graphs from scratch...')
            if n_jobs > 1:
                self.graphs = pmap(smiles_to_graph,
                                   self.smiles,
                                   n_jobs=n_jobs)
            else:
                self.graphs = []
                for i, s in enumerate(self.smiles):
                    if (i + 1) % log_every == 0:
                        print('Processing molecule {:d}/{:d}'.format(i+1, len(self)))
                    self.graphs.append(smiles_to_graph(s))

            # Keep only valid molecules
            self.valid_ids = []
            graphs = []
            failed_mols = []
            for i, g in enumerate(self.graphs):
                if g is not None:
                    self.valid_ids.append(i)
                    graphs.append(g)
                else:
                    failed_mols.append((i, self.smiles[i]))

            if error_log is not None:
                if len(failed_mols) > 0:
                    failed_ids, failed_smis = map(list, zip(*failed_mols))
                else:
                    failed_ids, failed_smis = [], []
                df = pd.DataFrame({'raw_id': failed_ids, 'smiles': failed_smis})
                df.to_csv(error_log, index=False)

            self.graphs = graphs
            _label_values = self.df[self.task_names].values
            # np.nan_to_num will also turn inf into a very large number
            self.labels = F.zerocopy_from_numpy(
                np.nan_to_num(_label_values).astype(np.float32))[self.valid_ids]
            valid_ids = torch.tensor(self.valid_ids)
            if init_mask:
                self.mask = F.zerocopy_from_numpy(
                    (~np.isnan(_label_values)).astype(np.float32))[self.valid_ids]
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'mask': self.mask,
                                    'valid_ids': valid_ids})
            else:
                self.mask = None
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'valid_ids': valid_ids})

        self.smiles = [self.smiles[i] for i in self.valid_ids]

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
            Labels of the datapoint for all tasks
        Tensor of dtype float32 and shape (T), optional
            Binary masks indicating the existence of labels for all tasks. This is only
            generated when ``init_mask`` is True in the initialization.
        """
        if self.mask is not None:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.smiles)

    def task_pos_weights(self, indices):
        """Get weights for positive samples on each task

        This should only be used when all tasks are binary classification.

        It's quite common that the number of positive samples and the number of
        negative samples are significantly different for binary classification.
        To compensate for the class imbalance issue, we can weight each datapoint
        in loss computation.

        In particular, for each task we will set the weight of negative samples
        to be 1 and the weight of positive samples to be the number of negative
        samples divided by the number of positive samples.

        Parameters
        ----------
        indices : 1D LongTensor
            The function will compute the weights on the data subset specified by
            the indices, e.g. the indices for the training set.

        Returns
        -------
        Tensor of dtype float32 and shape (T)
            Weight of positive samples on all tasks
        """
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = F.sum(self.labels[indices], dim=0)
        num_indices = F.sum(self.mask[indices], dim=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]

        return task_pos_weights
