# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dataset for JTVAE

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

def get_mol(smiles, kekulize=True):
    """Convert the SMILES string into an RDKit molecule object

    Parameters
    ----------
    smiles : str
        A SMILES string.
    kekulize : bool
        Whether to kekulize the molecule.

    Returns
    -------
    rdkit.Chem.rdchem.Mol or None
        A Kekulized RDKit molecule object if the input SMILES string is valid and None otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if kekulize:
        Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    """Convert an RDKit molecule object into a SMILES string.

    By default, the molecule is kekulized.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule.

    Returns
    -------
    str
        The SMILES string corresponding to the molecule.
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def decode_stereo(smiles_2d):
    """Get possible 3D SMILES by enumerating stereoisomers

    Parameters
    ----------
    smiles_2d : str
        A 2D SMILES string.

    Returns
    -------
    list of str
        List of possible 3D SMILES strings.
    """
    mol = Chem.MolFromSmiles(smiles_2d)
    # Get all possible stereoisomers for a molecule
    dec_isomers = list(EnumerateStereoisomers(mol))
    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(
        mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles_3d = [Chem.MolToSmiles(mol, isomericSmiles=True)
                for mol in dec_isomers]

    chiral_n = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms()
                if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]

    if len(chiral_n) > 0:
        for mol in dec_isomers:
            for idx in chiral_n:
                mol.GetAtomWithIdx(idx).SetChiralTag(
                    Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles_3d.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles_3d

def set_atommap(mol, num=0):
    """Set the atom map number for all atoms in the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule.
    num : int
        The atom map number to set for all atoms. If 0, it will
        clear the atom map.
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)
