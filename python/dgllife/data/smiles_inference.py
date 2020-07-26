# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dataset for inference on smiles

from rdkit import Chem

from ..utils.mol_to_graph import mol_to_bigraph

__all__ = ['UnlabeledSMILES']

class UnlabeledSMILES(object):
    """Construct a SMILES dataset without labels for inference.

    We will 1) Filter out invalid SMILES strings and record canonical SMILES strings
    for valid ones 2) Construct a DGLGraph for each valid one and feature its node/edge

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings
    mol_to_graph: callable, rdkit.Chem.rdchem.Mol -> DGLGraph
        A function turning an RDKit molecule object into a DGLGraph.
        Default to :func:`dgllife.utils.mol_to_bigraph`.
    node_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. Default to 1000.
    """
    def __init__(self, smiles_list, mol_to_graph=mol_to_bigraph, node_featurizer=None,
                 edge_featurizer=None, log_every=1000):
        super(UnlabeledSMILES, self).__init__()

        canonical_smiles = []
        mol_list = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol_list.append(mol)
            canonical_smiles.append(Chem.MolToSmiles(mol))

        self.smiles = canonical_smiles
        self.graphs = []
        for i, mol in enumerate(mol_list):
            if (i + 1) % log_every == 0:
                print('Processing molecule {:d}/{:d}'.format(i + 1, len(self)))
            self.graphs.append(mol_to_graph(mol, node_featurizer=node_featurizer,
                                            edge_featurizer=edge_featurizer))

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
        """
        return self.smiles[item], self.graphs[item]

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.smiles)
