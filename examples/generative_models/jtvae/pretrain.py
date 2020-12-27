# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Pre-trained an AE

import rdkit
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from dgllife.utils.jtvae import JTVAEVocab
from torch.utils.data import DataLoader

def main(args):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    if args.use_cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    dataset = JTVAEDataset(args.train_path)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=lambda x: x,
                            drop_last=True)

    vocab = JTVAEVocab(file_path=args.train_path)
    model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depth)

    # TODO: make this a function of model class
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)

    model = model.to(device)
    print("Model #Params: {:d}K".format(sum([x.nelement() for x in model.parameters()]) / 1000))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.gamma)

    for epoch in range(args.max_epoch):
        word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0

        for it, batch in enumerate(dataloader):
            for mol_tree in batch:
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
                        node.cand_mols.append(node.label_mol)

            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta=0)
            loss.backward()
            optimizer.step()

            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc

            if (it + 1) % args.print_iter == 0:
                word_acc = word_acc / args.print_iter * 100
                topo_acc = topo_acc / args.print_iter * 100
                assm_acc = assm_acc / args.print_iter * 100
                steo_acc = steo_acc / args.print_iter * 100

                print('KL: {:.1f}, Word: {:.2f}, Topo: {:.2f}, Assm: {:.2f}, Steo: {:.2f}'.format(
                    kl_div, word_acc, topo_acc, assm_acc, steo_acc))
                word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0
                sys.stdout.flush()

        scheduler.step()
        print("learning rate: {:.6f}".format(scheduler.get_lr()[0]))
        torch.save(model.state_dict(), args.save_path + "/model.iter-" + str(epoch))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-t', '--train-path', type=str,
                        help='Path to the training molecules, with one SMILES string a line')
    parser.add_argument('-s', '--save-path', type=str,
                        help='Directory to save model checkpoints')
    parser.add_argument('-b', '--batch-size', type=int, default=40,
                        help='Batch size')
    parser.add_argument('-w', '--hidden-size', type=int, default=200,
                        help='Hidden size')
    parser.add_argument('-l', '--latent-size', type=int, default=56,
                        help='Latent size')
    parser.add_argument('-d', '--depth', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('-g', '--gamma', type=float, default=0.9,
                        help='Multiplicative factor for learning rate decay')
    parser.add_argument('-me', '--max-epoch', type=int, default=3,
                        help='Maximum number of epochs for training')
    parser.add_argument('-nw', '--num-workers', type=int, default=4,
                        help='Number of subprocesses for data loading')
    parser.add_argument('-pi', '--print-iter', type=int, default=20,
                        help='Frequency for printing evaluation metrics')
    parser.add_argument('-cpu', '--use-cpu', action='store_true',
                        help='By default, the script uses GPU whenever available. '
                             'This flag enforces the use of CPU.')
    args = parser.parse_args()

    main(args)
