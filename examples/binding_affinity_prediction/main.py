# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from dgllife.utils.eval import Meter
from torch.utils.data import DataLoader

from utils import set_random_seed, load_dataset, collate, load_model


def rand_hyperparams():
    import numpy.random as nrd
    hyper_params = {}
    hyper_params['f_bond'] = nrd.randint(70,120)
    hyper_params['f_gather'] = nrd.randint(80,129)
    hyper_params['f_spatial'] = nrd.randint(hyper_params['f_gather'], 129)
    hyper_params['n_bond_conv_steps'] = nrd.randint(1,3)
    hyper_params['n_spatial_conv_steps'] = nrd.randint(1,2)
    hyper_params['wd'] = nrd.choice([1e-7, 1e-5])
    hyper_params['dropouts'] = [nrd.choice([0, 0.25, 0.4]) for i in range(3)]
    hyper_params['n_rows_fc'] = [nrd.choice([16])]
    hyper_params['max_num_neighbors'] = nrd.randint(3, 13)
    return hyper_params

def update_msg_from_scores(msg, scores):
    for metric, score in scores.items():
        msg += ', {} {:.4f}'.format(metric, score)
    return msg

def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    train_meter = Meter(args['train_mean'], args['train_std'])
    epoch_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        indices, ligand_mols, protein_mols, bg, labels = batch_data
        labels = labels.to(args['device'])
        if type(bg) == tuple: # for the case of PotentialNet
            bigraph_canonical, knn_graph = bg # unpack
            bigraph_canonical = bigraph_canonical.to(args['device'])
            knn_graph = knn_graph.to(args['device'])
            prediction = model(bigraph_canonical, knn_graph)
        else:
            bg = bg.to(args['device'])
            prediction = model(bg)
        loss = loss_criterion(prediction, (labels - args['train_mean']) / args['train_std'])
        epoch_loss += loss.data.item() * len(indices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels)
    avg_loss = epoch_loss / len(data_loader.dataset)
    if (args['num_epochs'] - epoch) <= 6: # print only the last 5 epochs
        total_scores = {metric: train_meter.compute_metric(metric, 'mean')
                        for metric in args['metrics']}
        msg = 'epoch {:d}/{:d}, training | loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], avg_loss)
        msg = update_msg_from_scores(msg, total_scores)
        print(msg)

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter(args['train_mean'], args['train_std'])
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            indices, ligand_mols, protein_mols, bg, labels = batch_data
            labels = labels.to(args['device'])
            if type(bg) == tuple: # for the case of PotentialNet
                bigraph_canonical, knn_graph = bg # unpack
                bigraph_canonical = bigraph_canonical.to(args['device'])
                knn_graph = knn_graph.to(args['device'])
                prediction = model(bigraph_canonical, knn_graph)
            else:
                bg = bg.to(args['device'])
                prediction = model(bg)
            eval_meter.update(prediction, labels)
    total_scores = {metric: eval_meter.compute_metric(metric, 'mean')
                    for metric in args['metrics']}
    return total_scores

def main(args):
    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(args['random_seed'])

    dataset, train_set, val_set, test_set = load_dataset(args)
    args['train_mean'] = train_set.labels_mean.to(args['device'])
    args['train_std'] = train_set.labels_std.to(args['device'])
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=args['shuffle'],
                              collate_fn=collate,
                              num_workers=8)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             shuffle=args['shuffle'],
                             collate_fn=collate,
                             num_workers=8)
    val_loader = DataLoader(dataset=val_set,
                             batch_size=args['batch_size'],
                             shuffle=args['shuffle'],
                             collate_fn=collate,
                             num_workers=8)

    model = load_model(args)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)

    test_scores = run_an_eval_epoch(args, model, test_loader)
    test_msg = update_msg_from_scores('test results', test_scores)
    print(test_msg)
    if args['frac_val'] > 0:
        val_scores = run_an_eval_epoch(args, model, val_loader)
        val_msg = update_msg_from_scores('validation results', val_scores)
        print(val_msg)

if __name__ == '__main__':
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Protein-Ligand Binding Affinity Prediction')
    parser.add_argument('-m', '--model', type=str, choices=['ACNN', 'PotentialNet'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['PDBBind_core_pocket_random', 'PDBBind_core_pocket_scaffold',
                                 'PDBBind_core_pocket_stratified', 'PDBBind_core_pocket_temporal',
                                 'PDBBind_refined_pocket_random', 'PDBBind_refined_pocket_scaffold',
                                 'PDBBind_refined_pocket_stratified', 'PDBBind_refined_pocket_temporal'],
                        help='Dataset to use')

    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    rand_hyper_search = True
    args['print_featurization'] = not rand_hyper_search
    if rand_hyper_search: # do hyperparameter search
        customized_hps = rand_hyperparams()
        args.update(customized_hps)
        for k, v in customized_hps.items():
            print(f'{k}: {v}')

    main(args)
    print('')
    print('')
