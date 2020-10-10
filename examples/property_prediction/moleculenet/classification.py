# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn

from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, EarlyStopping, Meter
from functools import partial
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import collate_molgraphs, load_model, predict

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

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
    if args['featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
        if args['edge_featurizer'] is not None:
            exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    exp_config.update({
        'n_tasks': args['n_tasks'],
        'model': args['model']
    })

    train_loader = DataLoader(dataset=train_set, batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    if args['pretrain']:
        args['num_epochs'] = 0
        if args['featurizer_type'] == 'pre_train':
            model = load_pretrained('{}_{}'.format(
                args['model'], args['dataset'])).to(args['device'])
        else:
            model = load_pretrained('{}_{}_{}'.format(
                args['model'], args['featurizer_type'], args['dataset'])).to(args['device'])
    else:
        model = load_model(exp_config).to(args['device'])
        loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                         weight_decay=exp_config['weight_decay'])
        stopper = EarlyStopping(patience=exp_config['patience'],
                                filename=args['result_path'] + '/model.pth',
                                metric=args['metric'])

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

    if not args['pretrain']:
        stopper.load_checkpoint(model)
    test_score = run_an_eval_epoch(args, model, test_loader)
    print('test {} {:.4f}'.format(args['metric'], test_score))

    with open(args['result_path'] + '/eval.txt', 'w') as f:
        if not args['pretrain']:
            f.write('Best val {}: {}\n'.format(args['metric'], stopper.best_score))
        f.write('Test {}: {}\n'.format(args['metric'], test_score))

if __name__ == '__main__':
    from argparse import ArgumentParser

    from utils import init_featurizer, mkdir_p, split_dataset, get_configure

    parser = ArgumentParser('Multi-label Binary Classification')
    parser.add_argument('-d', '--dataset', choices=['MUV', 'BACE', 'BBBP', 'ClinTox'],
                        help='Dataset to use')
    parser.add_argument('-mo', '--model', choices=['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP',
                                                   'gin_supervised_contextpred',
                                                   'gin_supervised_infomax',
                                                   'gin_supervised_edgepred',
                                                   'gin_supervised_masking'],
                        help='Model to use')
    parser.add_argument('-f', '--featurizer-type', choices=['canonical', 'attentivefp'],
                        help='Featurization for atoms (and bonds). This is required for models '
                             'other than gin_supervised_**.')
    parser.add_argument('-p', '--pretrain', action='store_true',
                        help='Whether to skip the training and evaluate the pre-trained model '
                             'on the test set (default: False)')
    parser.add_argument('-s', '--split', choices=['scaffold', 'random'], default='scaffold',
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 0.8,0.1,0.1)')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                        help='Maximum number of epochs for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 1000)')
    parser.add_argument('-nw', '--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-rp', '--result-path', type=str, default='classification_results',
                        help='Path to save training results (default: classification_results)')
    args = parser.parse_args().__dict__

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    args = init_featurizer(args)
    mkdir_p(args['result_path'])
    if args['dataset'] == 'MUV':
        from dgllife.data import MUV
        dataset = MUV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                      node_featurizer=args['node_featurizer'],
                      edge_featurizer=args['edge_featurizer'],
                      n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'BACE':
        from dgllife.data import BACE
        dataset = BACE(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                       node_featurizer=args['node_featurizer'],
                       edge_featurizer=args['edge_featurizer'],
                       n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'BBBP':
        from dgllife.data import BBBP
        dataset = BBBP(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                       node_featurizer=args['node_featurizer'],
                       edge_featurizer=args['edge_featurizer'],
                       n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'ClinTox':
        from dgllife.data import ClinTox
        dataset = ClinTox(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                          node_featurizer=args['node_featurizer'],
                          edge_featurizer=args['edge_featurizer'],
                          n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    else:
        raise ValueError('Unexpected dataset: {}'.format(args['dataset']))

    args['n_tasks'] = dataset.n_tasks
    train_set, val_set, test_set = split_dataset(args, dataset)
    exp_config = get_configure(args['model'], args['featurizer_type'], args['dataset'])
    main(args, exp_config, train_set, val_set, test_set)
