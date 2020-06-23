# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from dgllife.model import load_pretrained
from dgllife.utils import EarlyStopping, Meter
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import set_random_seed, load_dataset_for_classification, collate_molgraphs, load_model

def predict(args, model, bg):
    node_feats = bg.ndata.pop(args['node_data_field']).to(args['device'])
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata.pop(args['edge_data_field']).to(args['device'])
        return model(bg, node_feats, edge_feats)
    else:
        return model(bg, node_feats)

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
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(logits, labels, masks)
    train_score = np.mean(train_meter.compute_metric(args['metric_name']))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric_name'], train_score))

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            logits = predict(args, model, bg)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric_name']))

def main(args):
    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(args['random_seed'])

    # Interchangeable with other datasets
    dataset, train_set, val_set, test_set = load_dataset_for_classification(args)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)

    if args['pre_trained']:
        args['num_epochs'] = 0
        model = load_pretrained(args['exp'])
    else:
        args['n_tasks'] = dataset.n_tasks
        model = load_model(args)
        loss_criterion = BCEWithLogitsLoss(pos_weight=dataset.task_pos_weights.to(args['device']),
                                           reduction='none')
        optimizer = Adam(model.parameters(), lr=args['lr'])
        stopper = EarlyStopping(patience=args['patience'])
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric_name'],
            val_score, args['metric_name'], stopper.best_score))
        if early_stop:
            break

    if not args['pre_trained']:
        stopper.load_checkpoint(model)
    test_score = run_an_eval_epoch(args, model, test_loader)
    print('test {} {:.4f}'.format(args['metric_name'], test_score))

if __name__ == '__main__':
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='MoleculeNet')
    parser.add_argument('-m', '--model', type=str, choices=['GCN', 'GAT', 'Weave'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['Tox21'], default='Tox21',
                        help='Dataset to use')
    parser.add_argument('-p', '--pre-trained', action='store_true',
                        help='Whether to skip training and use a pre-trained model')
    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    main(args)
