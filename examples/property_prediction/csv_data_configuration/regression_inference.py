# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from dgllife.data import UnlabeledSMILES
from dgllife.utils import CanonicalAtomFeaturizer

def main(args):
    return NotImplementedError

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Inference for (Multitask) Regression')
    parser.add_argument('-f', '--file-path', type=str, required=True,
                        help='Path to a .csv/.txt file of SMILES strings')
    parser.add_argument('-sc', '--smiles-column', type=str,
                        help='Header for the SMILES column in the CSV file, can be '
                             'omitted if the input file is a .txt file or the .csv '
                             'file only has one column of SMILES strings')
    parser.add_argument('-tp', '--train-result-path', type=str, default='regression_results',
                        help='Path to the saved training results, which will be used for '
                             'loading the trained model and related configurations')
    parser.add_argument('-ip', '--inference-result-path', type=str, default='regression_inference_results',
                        help='Path to save the inference results')
    parser.add_argument('-t', '--task-names', default=None, type=str,
                        help='Task names for saving model predictions in the CSV file to output, '
                             'which should be the same as the ones used for training. If not '
                             'specified, we will simply use task1, task2, ...')
    args = parser.parse_args().__dict__

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    if args['file_path'].endswith('.csv'):
        import pandas
        df = pandas.read_csv(args['file_path'])
        if args['smiles_column'] is not None:
            smiles = df[args['smiles_column']].tolist()
        else:
            assert len(df.columns) == 1, 'The CSV file has more than 1 columns and ' \
                                         '-sc (smiles-column) needs to be specified.'
            smiles = df[df.columns[0]].tolist()
    elif args['file_path'].endswith('.txt'):
        from dgllife.utils import load_smiles_from_txt
        smiles = load_smiles_from_txt(args['file_path'])
    else:
        raise ValueError('Expect the input data file to be a .csv or a .txt file, '
                         'got {}'.format(args['file_path']))
    args['smiles'] = smiles

    main(args)
