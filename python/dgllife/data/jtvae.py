# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Dataset for JTVAE

from dgl.data.utils import get_download_dir, _get_dgl_url, download, extract_archive
from torch.utils.data import Dataset

from ..utils.jtvae.mol_tree import MolTree

__all__ = ['JTVAEDataset',
           'JTVAEZINC']

class JTVAEDataset(Dataset):
    """Dataset for JTVAE

    Parameters
    ----------
    data_file : str
        Path to a file of SMILES strings, with one SMILES string a line.
    cache_tree : bool
        Whether to cache the trees to speed up data loading or always construct trees on the fly.
    """
    def __init__(self, data_file, cache_tree=False):
        with open(data_file, 'r') as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]
        self.cache_tree = cache_tree
        if cache_tree:
            self.trees = [None for _ in range(len(self))]

    def __len__(self):
        """Get the size of the dataset

        Returns
        -------
        int
            Number of molecules in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Get a datapoint corresponding to the index.

        Parameters
        ----------
        idx : int
            ID for the datapoint.

        Returns
        -------
        MolTree
            MolTree corresponding to the datapoint.
        """
        if self.cache_tree and self.trees[idx] is not None:
            return self.trees[idx]
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()

        if self.cache_tree:
            self.trees[idx] = mol_tree

        return mol_tree

class JTVAEZINC(JTVAEDataset):
    """A ZINC subset used in JTVAE

    Parameters
    ----------
    subset : train
        TODO: check
        The subset to use, which can be one of 'train', 'val', and 'test'.
    cache_tree : bool
        Whether to cache the trees to speed up data loading or always construct trees on the fly.
    """
    def __init__(self, subset, cache_tree=False):
        # TODO: check subset
        dir = get_download_dir()
        _url = _get_dgl_url('dataset/jtnn.zip')
        zip_file_path = '{}/jtnn.zip'.format(dir)
        download(_url, path=zip_file_path)
        extract_archive(zip_file_path, '{}/jtnn'.format(dir))

        if subset == 'train':
            super(JTVAEZINC, self).__init__(data_file = '{}/jtnn/{}.txt'.format(dir, subset),
                                            cache_tree=cache_tree)
        else:
            raise NotImplementedError('Unexpected subset: {}'.format(subset))
