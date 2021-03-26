# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained generative models.

import os

from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive

__all__ = ['generative_url',
           'create_generative_model']

generative_url = {
    'DGMG_ChEMBL_canonical': 'pre_trained/dgmg_ChEMBL_canonical.pth',
    'DGMG_ChEMBL_random': 'pre_trained/dgmg_ChEMBL_random.pth',
    'DGMG_ZINC_canonical': 'pre_trained/dgmg_ZINC_canonical.pth',
    'DGMG_ZINC_random': 'pre_trained/dgmg_ZINC_random.pth',
    'JTNN_ZINC': 'pre_trained/JTNN_ZINC.pth'
}

try:
    # Things requiring RDKit
    from rdkit import Chem
    from ...model import DGMG, DGLJTNNVAE
except ImportError:
    pass

def create_generative_model(model_name):
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name for the model.

    Returns
    -------
    Created model
    """
    if model_name.startswith('DGMG'):
        if model_name.startswith('DGMG_ChEMBL'):
            atom_types = ['O', 'Cl', 'C', 'S', 'F', 'Br', 'N']
        elif model_name.startswith('DGMG_ZINC'):
            atom_types = ['Br', 'S', 'C', 'P', 'N', 'O', 'F', 'Cl', 'I']
        bond_types = [Chem.rdchem.BondType.SINGLE,
                      Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE]

        return DGMG(atom_types=atom_types,
                    bond_types=bond_types,
                    node_hidden_size=128,
                    num_prop_rounds=2,
                    dropout=0.2)

    elif model_name == "JTNN_ZINC":
        default_dir = get_download_dir()
        vocab_file = '{}/jtvae/{}.txt'.format(default_dir, 'vocab')
        if not os.path.exists(vocab_file):
            zip_file_path = '{}/jtvae.zip'.format(default_dir)
            download(_get_dgl_url('dataset/jtvae.zip'), path=zip_file_path)
            extract_archive(zip_file_path, '{}/jtvae'.format(default_dir))
        return DGLJTNNVAE(vocab_file=vocab_file,
                          depth=3,
                          hidden_size=450,
                          latent_size=56)

    else:
        return None
