# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import Meter, smiles_to_bigraph, CanonicalAtomFeaturizer, EarlyStopping
from hyperopt import fmin, tpe
from shutil import copyfile
from torch.optim import Adam
from torch.utils.data import DataLoader

from hyper import init_hyper_space
from utils import get_configure, mkdir_p, init_trial_path, \
    split_dataset, collate_molgraphs, load_model, predict, init_featurizer

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        labels, masks = labels.to(args['device']), masks.to(args['device'])
        logits = predict(args, model, bg)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)
        if batch_id % args['print_every'] == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
    train_score = np.mean(train_meter.compute_metric(args['metric']))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], train_score))

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            logits = predict(args, model, bg)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric']))

def main(args, exp_config, train_set, val_set, test_set):
    # Record settings
    exp_config.update({
        'model': args['model'],
        'in_feats': args['node_featurizer'].feat_size(),
        'n_tasks': args['n_tasks'],
        'atom_featurizer_type': args['atom_featurizer_type'],
        'bond_featurizer_type': args['bond_featurizer_type']
    })

    # Set up directory for saving results
    args = init_trial_path(args)

    train_loader = DataLoader(dataset=train_set, batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    model = load_model(exp_config).to(args['device'])

    loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                     weight_decay=exp_config['weight_decay'])
    stopper = EarlyStopping(mode=args['early_stop_mode'],
                            patience=exp_config['patience'],
                            filename=args['trial_path'] + '/model.pth')

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'],
            val_score, args['metric'], stopper.best_score))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_score = run_an_eval_epoch(args, model, test_loader)
    print('test {} {:.4f}'.format(args['metric'], test_score))

    with open(args['trial_path'] + '/eval.txt', 'w') as f:
        f.write('Best val {}: {}\n'.format(args['metric'], stopper.best_score))
        f.write('Test {}: {}\n'.format(args['metric'], test_score))

    with open(args['trial_path'] + '/configure.json', 'w') as f:
        json.dump(exp_config, f, indent=2)

    return args['trial_path'], stopper.best_score

def bayesian_optimization(args, train_set, val_set, test_set):
    # Run grid search
    results = []

    candidate_hypers = init_hyper_space(args['model'])

    def objective(hyperparams):
        configure = deepcopy(args)
        trial_path, val_metric = main(configure, hyperparams, train_set, val_set, test_set)

        if args['metric'] in ['roc_auc_score']:
            # Maximize ROCAUC is equivalent to minimize the negative of it
            val_metric_to_minimize = -1 * val_metric
        else:
            val_metric_to_minimize = val_metric

        results.append((trial_path, val_metric_to_minimize))

        return val_metric_to_minimize

    fmin(objective, candidate_hypers, algo=tpe.suggest, max_evals=args['num_evals'])
    results.sort(key=lambda tup: tup[1])
    best_trial_path, best_val_metric = results[0]

    return best_trial_path

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Multi-label Binary Classification')
    parser.add_argument('-c', '--csv-path', type=str, required=True,
                        help='Path to a csv file for loading a dataset')
    parser.add_argument('-sc', '--smiles-column', type=str, required=True,
                        help='Header for the SMILES column in the CSV file')
    parser.add_argument('-t', '--task-names', default=None, type=str,
                        help='Header for the tasks to model. If None, we will model '
                             'all the columns except for the smiles_column in the CSV file. '
                             '(default: None)')
    parser.add_argument('-s', '--split', choices=['scaffold'], default='scaffold',
                        help='Dataset splitting method')
    parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of the dataset used for training, validation and test')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score'], default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-mo', '--model', choices=['GCN', 'GAT'], default='GCN',
                        help='Model to use (default: GCN)')
    parser.add_argument('-a', '--atom-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for atoms (default: CanonicalAtomFeaturizer)')
    parser.add_argument('-b', '--bond-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for bonds (default: canonical)')
    parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                        help='Maximum number of epochs allowed for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 1000)')
    parser.add_argument('-nw', '--num-workers', type=int, default=1,
                        help='Number of processes for data loading (default: 1)')
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-p', '--result-path', type=str, default='classification_results',
                        help='Path to save training results (default: classification_results)')
    parser.add_argument('-ne', '--num-evals', type=int, default=None,
                        help='Number of trials for hyperparameter search (default: None)')
    args = parser.parse_args().__dict__

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    if args['task_names'] is not None:
        args['task_names'] = args['task_names'].split(',')

    if args['metric'] == 'roc_auc_score':
        args['early_stop_mode'] = 'higher'

    args = init_featurizer(args)
    df = pd.read_csv(args['csv_path'])
    mkdir_p(args['result_path'])
    dataset = MoleculeCSVDataset(df=df,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=args['node_featurizer'],
                                 edge_featurizer=args['edge_featurizer'],
                                 smiles_column=args['smiles_column'],
                                 cache_file_path=args['result_path'] + '/graph.bin',
                                 task_names=args['task_names'])
    args['n_tasks'] = dataset.n_tasks
    train_set, val_set, test_set = split_dataset(args, dataset)

    if args['num_evals'] is not None:
        assert args['num_evals'] > 0, 'Expect the number of hyperparameter search trials to ' \
                                      'be greater than 0, got {:d}'.format(args['num_evals'])
        print('Start hyperparameter search with Bayesian '
              'optimization for {:d} trials'.format(args['num_evals']))
        trial_path = bayesian_optimization(args, train_set, val_set, test_set)
    else:
        print('Use the manually specified hyperparameters')
        exp_config = get_configure(args['model'])
        main(args, exp_config, train_set, val_set, test_set)
        trial_path = args['result_path'] + '/1'

    # Copy final
    copyfile(trial_path + '/model.pth', args['result_path'] + '/model.pth')
    copyfile(trial_path + '/configure.json', args['result_path'] + '/configure.json')
    copyfile(trial_path + '/eval.txt', args['result_path'] + '/eval.txt')
