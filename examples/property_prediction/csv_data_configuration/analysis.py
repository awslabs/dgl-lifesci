# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

if __name__ == '__main__':
    import pandas

    from argparse import ArgumentParser
    from dgllife.utils import analyze_mols

    from utils import mkdir_p

    parser = ArgumentParser('Dataset analysis')
    parser.add_argument('-c', '--csv-path', type=str, required=True,
                        help='Path to a csv file for loading a dataset')
    parser.add_argument('-sc', '--smiles-column', type=str, required=True,
                        help='Header for the SMILES column in the CSV file')
    parser.add_argument('-np', '--num-processes', type=int, default=1,
                        help='Number of processes to use for analysis')
    parser.add_argument('-p', '--path', type=str, default='analysis_results',
                        help='Path to export analysis results')
    args = parser.parse_args().__dict__

    mkdir_p(args['path'])

    df = pandas.read_csv(args['csv_path'])
    analyze_mols(smiles=df[args['smiles_column']].tolist(),
                 num_processes=args['num_processes'],
                 path_to_export=args['path'])
