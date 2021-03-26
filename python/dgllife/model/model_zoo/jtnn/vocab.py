# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=C0111, C0103, E1101, W0611, W0612

import copy
import rdkit.Chem as Chem

def get_slots(smiles):
    """
    Parameters
    ----------
    smiles : str
        A SMILES string.

    Returns
    -------
    list of 3-tuples
        Each tuple consists of the symbol, formal charge and the total
        number of hydrogen atoms attached of each atom.
    """
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs())
            for atom in mol.GetAtoms()]

class Vocab(object):
    """Vocabulary for JTVAE

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings for vocabulary.
    """
    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles):
        """Return the index corresponding to the SMILES string"""
        return self.vmap[smiles]

    def get_smiles(self, idx):
        """Return the SMILES string corresponding to the index"""
        return self.vocab[idx]

    def get_slots(self, idx):
        """Get the slots for the SMILES corresponding to the index"""
        return copy.deepcopy(self.slots[idx])

    def size(self):
        """Return the size of the vocabulary"""
        return len(self.vocab)
