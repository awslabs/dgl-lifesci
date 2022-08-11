# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pandas as pd
import torch

from dgllife.data import UnlabeledSMILES
from dgllife.utils import MolToBigraph
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import mkdir_p, collate_molgraphs_unlabeled, load_model, predict, init_featurizer

def main(args):
    mol_to_g = MolToBigraph(add_self_loop=True,
                            node_featurizer=args['node_featurizer'],
                            edge_featurizer=args['edge_featurizer'])
    dataset = UnlabeledSMILES(args['smiles'], mol_to_graph=mol_to_g)
    dataloader = DataLoader(dataset, batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs_unlabeled, num_workers=args['num_workers'])
    model = load_model(args).to(args['device'])
    checkpoint = torch.load(args['train_result_path'] + '/model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    smiles_list = []
    predictions = []

    with torch.no_grad():
        for batch_id, batch_data in enumerate(tqdm(dataloader, desc="Iteration")):
            batch_smiles, bg = batch_data
            smiles_list.extend(batch_smiles)
            batch_pred = predict(args, model, bg)
            predictions.append(batch_pred.detach().cpu())

    predictions = torch.cat(predictions, dim=0)

    output_data = {'canonical_smiles': smiles_list}
    if args['task_names'] is None:
        args['task_names'] = ['task_{:d}'.format(t) for t in range(1, args['n_tasks'] + 1)]
    else:
        args['task_names'] = args['task_names'].split(',')
    for task_id, task_name in enumerate(args['task_names']):
        output_data[task_name] = predictions[:, task_id]
    df = pd.DataFrame(output_data)
    df.to_csv(args['inference_result_path'] + '/prediction.csv', index=False)

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
    parser.add_argument('-nw', '--num-workers', type=int, default=1,
                        help='Number of processes for data loading (default: 1)')
    args = parser.parse_args().__dict__

    # Load configuration
    with open(args['train_result_path'] + '/configure.json', 'r') as f:
        args.update(json.load(f))

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    if args['file_path'].endswith('.csv') or args['file_path'].endswith('.csv.gz'):
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
    args = init_featurizer(args)
    # Handle directories
    mkdir_p(args['inference_result_path'])
    assert os.path.exists(args['train_result_path']), \
        'The path to the saved training results does not exist.'

    main(args)
