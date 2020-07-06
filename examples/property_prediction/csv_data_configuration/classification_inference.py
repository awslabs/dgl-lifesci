# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Inference for Multi-label Binary Classification')
    parser.add_argument('-f', '--file-path', type=str, required=True,
                        help='Path to a .csv/.txt file of SMILES strings')
    parser.add_argument('-sc', '--smiles-column', type=str,
                        help='Header for the SMILES column in the CSV file, can be '
                             'omitted if the input file is a .txt file or the .csv '
                             'file only has one column of SMILES strings')
    parser.add_argument('-tp', '--train-result-path', type=str, default='classification_results',
                        help='Path to the saved training results, which will be used for '
                             'loading the trained model and related configurations')
    parser.add_argument('-ip', '--inference-result-path', type=str, default='classification_inference_results',
                        help='Path to save the inference results')
    parser.add_argument('-t', '--task-names', default=None, type=str,
                        help='Task names for saving model predictions in the CSV file to output, '
                             'which should be the same as the ones used for training. If not '
                             'specified, we will simply use task1, task2, ...')
    parser.add_argument('-s', '--soft-classification', action='store_true', default=False,
                        help='By default we will perform hard classification with binary labels. '
                             'This flag allows performing soft classification instead.')
