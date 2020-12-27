# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Test for utils in JTVAE

import os

def remove_file(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass

def test_vocab():
    from dgllife.utils.jtvae import JTVAEVocab

    vocab = JTVAEVocab()
    assert vocab.get_smiles(0) == 'C1=[NH+]C=[NH+]CC1'
    assert vocab.get_index('C1=[NH+]C=[NH+]CC1') == 0

    tmp_file = 'tmp.txt'
    with open(tmp_file, 'w') as f:
        f.write('CCO\n')
        f.write('C1=CC2=CC=CC=CC2=C1\n')
        f.write('O=C(O)/C=C/C(=O)O\n')
        f.write('N[C@@H](C)C(=O)O\n')
    vocab = JTVAEVocab(tmp_file)
    assert set(vocab.vocab) == \
           set(['C=C', 'C1=CCC=C1', 'C', 'C=O', 'CN', 'C1=CC=CCC=C1', 'CO', 'CC'])
    remove_file(tmp_file)

if __name__ == '__main__':
    test_vocab()
