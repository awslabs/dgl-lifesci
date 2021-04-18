# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Vocab in JTVAE

from copy import  deepcopy
from rdkit import Chem

from .mol_tree import MolTree

__all__ = ['JTVAEVocab']

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(),
             atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class JTVAEVocab(object):
    """Vocabulary for JTVAE.

    Parameters
    ----------
    file_path : str
        Path to the vocabulary file, which is a txt file of SMILES strings,
        one SMILES string a line.
    """
    def __init__(self, file_path=None):
        if file_path is None:
            from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive

            default_dir = get_download_dir()
            vocab_file = '{}/jtvae/vocab.txt'.format(default_dir)
            zip_file_path = '{}/jtvae.zip'.format(default_dir)
            download(_get_dgl_url('dataset/jtvae.zip'), path=zip_file_path, overwrite=False)
            extract_archive(zip_file_path, '{}/jtvae'.format(default_dir))

            with open(vocab_file, 'r') as f:
                self.vocab = [x.strip("\r\n ") for x in f]
        else:
            # Prepare a vocabulary from scratch
            vocab = set()
            with open(file_path, 'r') as f:
                for line in f:
                    smiles = line.split()[0]
                    mol = MolTree(smiles)
                    for i in mol.nodes_dict:
                        vocab.add(mol.nodes_dict[i]['smiles'])
            self.vocab = list(vocab)

        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles):
        """Get the index for a vocabulary token.

        Parameters
        ----------
        smiles : str
            The SMILES string for a vocabulary token.

        Returns
        -------
        int
            The ID for the token.
        """
        return self.vmap[smiles]

    def get_smiles(self, idx):
        """Get the SMILES string corresponding to the index.

        Parameters
        ----------
        idx : int
            The ID for a vocabulary token.

        Returns
        -------
        str
            The vocabulary token corresponding to the index.
        """
        return self.vocab[idx]

    def get_slots(self, idx):
        """Get 3-tuples of atom symbol, formal charge and total number of hydrogen atoms for
        all atoms in the vocabulary token corresponding to the index.

        Parameters
        ----------
        idx : int
            The ID for a vocabulary token.

        Returns
        -------
        list of 3-tuples
            Each tuple consists of symbol, formal charge and total number of hydrogen atoms for
            an atom in the vocabulary token.
        """
        return deepcopy(self.slots[idx])

    def size(self):
        """Get the size of the vocabulary.

        Returns
        -------
        int
            The vocabulary size.
        """
        return len(self.vocab)
