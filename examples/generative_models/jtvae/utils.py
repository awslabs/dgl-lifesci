# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import rdkit
from dgl.data.utils import get_download_dir

def worker_init_fn(id_):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

def get_vocab_file(vocab):
    """Get the path to the vocabulary file.

    Parameters
    ----------
    vocab : str
        * If 'zinc' or 'guacamol', it will return the path to the
          vocabulary file for the corresponding dataset.
        * Otherwise, it should be already the path to a vocabulary file.

    Returns
    -------
    str
        The path to a vocabulary file.
    """
    dir = get_download_dir()
    if vocab == 'zinc':
        vocab_file = '{}/jtnn/vocab.txt'.format(dir)
    elif vocab == 'guacamol':
        vocab_file = '{}/jtnn/vocab_guacamol.txt'.format(dir)
    else:
        vocab_file = vocab

    return vocab_file
