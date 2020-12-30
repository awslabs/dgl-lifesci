# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Dataset for JTVAE

import dgl
import dgl.function as fn
import torch

from dgl.data.utils import get_download_dir, _get_dgl_url, download, extract_archive
from functools import partial
from rdkit import Chem
from torch.utils.data import Dataset

from ..utils.featurizers import BaseAtomFeaturizer, ConcatFeaturizer, atom_type_one_hot, \
    atom_degree_one_hot, atom_formal_charge_one_hot, atom_chiral_tag_one_hot, atom_is_aromatic, \
    BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring, bond_stereo_one_hot
from ..utils.jtvae.chemutils import get_mol
from ..utils.jtvae.mol_tree import MolTree
from ..utils.mol_to_graph import mol_to_bigraph

__all__ = ['JTVAEDataset',
           'JTVAEZINC',
           'JTVAECollator']

def get_atom_featurizer_enc():
    """Get the atom featurizer for encoding.

    Returns
    -------
    BaseAtomFeaturizer
        The atom featurizer for encoding.
    """
    featurizer = BaseAtomFeaturizer({'x': ConcatFeaturizer([
        partial(atom_type_one_hot,
                allowable_set=['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                               'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn'],
                encode_unknown=True),
        partial(atom_degree_one_hot, allowable_set=[0, 1, 2, 3, 4], encode_unknown=True),
        partial(atom_formal_charge_one_hot, allowable_set=[-1, -2, 1, 2],
                encode_unknown=True),
        partial(atom_chiral_tag_one_hot,
                allowable_set=[Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                               Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                               Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW],
                encode_unknown=True),
        atom_is_aromatic
    ])})
    return featurizer

def get_bond_featurizer_enc():
    """Get the bond featurizer for encoding.

    Returns
    -------
    BaseBondFeaturizer
        The bond featurizer for encoding.
    """
    featurizer = BaseBondFeaturizer({'x': ConcatFeaturizer([
        bond_type_one_hot,
        bond_is_in_ring,
        partial(bond_stereo_one_hot,
                allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                               Chem.rdchem.BondStereo.STEREOANY,
                               Chem.rdchem.BondStereo.STEREOZ,
                               Chem.rdchem.BondStereo.STEREOE,
                               Chem.rdchem.BondStereo.STEREOCIS],
                encode_unknown=True)
    ])})
    return featurizer

class JTVAEDataset(Dataset):
    """Dataset for JTVAE

    Parameters
    ----------
    data_file : str
        Path to a file of SMILES strings, with one SMILES string a line.
    vocab : JTVAEVocab
        Vocabulary for JTVAE.
    cache : bool
        Whether to cache the trees to speed up data loading or always construct trees on the fly.
    """
    def __init__(self, data_file, vocab, cache=False):
        with open(data_file, 'r') as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]
        self.vocab = vocab
        self.cache = cache
        if cache:
            self.trees = [None for _ in range(len(self))]
            self.mol_graphs = [None for _ in range(len(self))]

        self.atom_featurizer_enc = get_atom_featurizer_enc()
        self.bond_featurizer_enc = get_bond_featurizer_enc()

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
        if self.cache and self.trees[idx] is not None:
            return self.trees[idx], self.mol_graphs[idx]
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()

        wid = [self.vocab.get_index(mol_tree.nodes_dict[i]['smiles'])
               for i in mol_tree.nodes_dict]
        mol_tree.graph.ndata['wid'] = torch.LongTensor(wid)

        # Construct molecular graphs
        mol = get_mol(smiles)
        mol_graph = mol_to_bigraph(mol,
                                   node_featurizer=self.atom_featurizer_enc,
                                   edge_featurizer=self.bond_featurizer_enc,
                                   canonical_atom_order=False)
        mol_graph.apply_edges(fn.copy_u('x', 'src'))
        mol_graph.edata['x'] = torch.cat(
            [mol_graph.edata.pop('src'), mol_graph.edata['x']], dim=1)

        if self.cache:
            self.trees[idx] = mol_tree
            self.mol_graphs[idx] = mol_graph

        return mol_tree, mol_graph

class JTVAEZINC(JTVAEDataset):
    """A ZINC subset used in JTVAE

    Parameters
    ----------
    subset : train
        TODO: check
        The subset to use, which can be one of 'train', 'val', and 'test'.
    vocab : JTVAEVocab
        Vocabulary for JTVAE.
    cache : bool
        Whether to cache the trees to speed up data loading or always construct trees on the fly.
    """
    def __init__(self, subset, vocab, cache=False):
        # TODO: check subset
        dir = get_download_dir()
        _url = _get_dgl_url('dataset/jtvae.zip')
        zip_file_path = '{}/jtvae.zip'.format(dir)
        download(_url, path=zip_file_path)
        extract_archive(zip_file_path, '{}/jtvae'.format(dir))

        if subset == 'train':
            super(JTVAEZINC, self).__init__(data_file = '{}/jtvae/{}.txt'.format(dir, subset),
                                            vocab=vocab, cache=cache)
        else:
            raise NotImplementedError('Unexpected subset: {}'.format(subset))

class JTVAECollator(object):
    """Collate function for JTVAE.

    Parameters
    ----------
    training : bool
        Whether the collate function is for training or not.
    """
    def __init__(self, training=True):
        self.training = training

    def __call__(self, data):
        """Batch multiple datapoints

        Parameters
        ----------
        data : list of tuples
            Multiple datapoints.

        Returns
        -------
        list of MolTree
            Junction trees for a batch of datapoints.
        DGLGraph
            Batched graph for the junction trees.
        DGLGraph
            Batched graph for the molecular graphs.
        """
        batch_trees, batch_mol_graphs = map(list, zip(*data))
        batch_tree_graphs = dgl.batch([tree.graph for tree in batch_trees])
        batch_mol_graphs = dgl.batch(batch_mol_graphs)

        return batch_trees, batch_tree_graphs, batch_mol_graphs
