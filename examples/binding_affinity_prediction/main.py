# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn

from dgllife.utils.eval import Meter
from torch.utils.data import DataLoader

from utils import set_random_seed, load_dataset, collate, load_model, rand_hyperparams


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
        else: #ACNN
            bg = bg.to(args['device'])
            prediction = model(bg)
        loss = loss_criterion(prediction, (labels - args['train_mean']) / args['train_std'])
        epoch_loss += loss.data.item() * len(indices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels)
    avg_loss = epoch_loss / len(data_loader.dataset)
    # if (args['num_epochs'] - epoch) <= 6: # print only the last 5 epochs
    total_scores = {metric: train_meter.compute_metric(metric, 'mean')
                    for metric in args['metrics']}
    msg = 'epoch {:d}/{:d}, training | loss {:.4f}'.format(
        epoch + 1, args['num_epochs'], avg_loss)
    msg = update_msg_from_scores(msg, total_scores)
    print(msg)
    return total_scores

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
    torch.multiprocessing.set_sharing_strategy('file_system') # test
    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(args['random_seed'])

    _, train_set, val_set, test_set = load_dataset(args) 
    args['train_mean'] = train_set.labels_mean.to(args['device'])
    args['train_std'] = train_set.labels_std.to(args['device'])

    if args['test_on_core']:   # always test on core set
        test_loader_dict = {'subset': 'core', 'frac_train':0, 'frac_val': 0, 'frac_test': 1}
        args.update(test_loader_dict)
        _, _, _, test_set = load_dataset(args)

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

    n_trials = args['num_trials']
    n_epochs = args['num_epochs']
    trial_train_r2, trial_val_r2, trial_test_r2 = np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials)
    for trial in range(n_trials):
        print(f'\n Running trial {trial + 1}/{n_trials}: \n')
        model = load_model(args)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
        model.to(args['device'])
        train_r2, val_r2, test_r2 = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)
        for epoch in range(n_epochs):
            train_scores = run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)
            train_r2[epoch] = train_scores['r2']
            if len(val_set) > 0:
                val_scores = run_an_eval_epoch(args, model, val_loader)
                val_msg = update_msg_from_scores('validation results', val_scores)
                print(val_msg)
                val_r2[epoch] = val_scores['r2']
            if len(test_set) > 0:
                test_scores = run_an_eval_epoch(args, model, test_loader)
                test_msg = update_msg_from_scores('test results', test_scores)
                print(test_msg)
                test_r2[epoch] = test_scores['r2']
            print('')
        # save model r2 at each epoch
        if args['save_r2']:
            save_path = args['save_r2'] + "/{}_{}_{}_{}_trial{}.npz".format(args['model'], args['version'], args['subset'], args['split'], trial+1)
            np.savez(save_path, train_r2=train_r2, val_r2=val_r2, test_r2=test_r2)

        # save results on the epoch with best test r2
        best_epoch = np.argmax(test_r2)
        trial_train_r2[trial] = train_r2[best_epoch]
        trial_val_r2[trial] = val_r2[best_epoch]
        trial_test_r2[trial] = test_r2[best_epoch]
        print('Best test epoch: ', best_epoch + 1)

    print(f'\n Finished {n_trials} trials.')
    print('Best train R2 mean: ', trial_train_r2.mean(), ', standard deviation: ', trial_train_r2.std())
    print('Best validation R2 mean: ', trial_val_r2.mean(), ', standard deviation: ', trial_val_r2.std())
    print('Best test R2 mean: ', trial_test_r2.mean(), ', standard deviation: ', trial_test_r2.std())


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
                                 'PDBBind_refined_pocket_stratified', 'PDBBind_refined_pocket_temporal',
                                 'PDBBind_refined_pocket_structure', 'PDBBind_refined_pocket_sequence'],
                        help='Data subset to use')
    parser.add_argument('-v', '--version', type=str, choices=['v2007', 'v2015'])
    parser.add_argument('--save_r2', type=str, default='', help='path to save r2 at each epoch, default not save')
    parser.add_argument('-t', '--num_trials', type=int, default=1)
    parser.add_argument('--test_on_core', type=bool, default=True, 
        help='whether to use the whole core set as test set, default True')

    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    if args['split']=='sequence' or args['split']=='structure':
        args['version'] = 'v2007' 
        args['test_on_core'] = False

    rand_hyper_search = False
    if rand_hyper_search: # randomly initialize hyperparameters
        customized_hps = rand_hyperparams()
        args.update(customized_hps)
    for k, v in args.items():
        print(f'{k}: {v}')

    print('')
    main(args)
    print('')
