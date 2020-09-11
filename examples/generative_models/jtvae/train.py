# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import rdkit
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from dgllife.model import DGLJTNNVAE
from torch.utils.data import DataLoader

from .datautils import JTNNDataset, JTNNCollator

torch.multiprocessing.set_sharing_strategy('file_system')

def worker_init_fn(id_):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

worker_init_fn(None)

parser = argparse.ArgumentParser(description="Training for JTNN",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--train", dest="train", default='train', help='Training file name')
parser.add_argument("-v", "--vocab", dest="vocab", default='zinc', help='Vocab file name')
parser.add_argument("-s", "--save_dir", dest="save_path", default='./',
                    help="Path to save checkpoint models, default to be current working directory")
parser.add_argument("-m", "--model", dest="model_path", default=None,
                    help="Path to load pre-trained model")
parser.add_argument("-b", "--batch", dest="batch_size", default=40,
                    help="Batch size")
parser.add_argument("-w", "--hidden", dest="hidden_size", default=200,
                    help="Size of representation vectors")
parser.add_argument("-l", "--latent", dest="latent_size", default=56,
                    help="Latent Size of node(atom) features and edge(atom) features")
parser.add_argument("-d", "--depth", dest="depth", default=3,
                    help="Depth of message passing hops")
parser.add_argument("-z", "--beta", dest="beta", default=1.0,
                    help="Coefficient of KL Divergence term")
parser.add_argument("-q", "--lr", dest="lr", default=1e-3,
                    help="Learning Rate")
args = parser.parse_args()

dataset = JTNNDataset(data=args.train, vocab=args.vocab, training=True)
vocab_file = dataset.vocab_file

batch_size = int(args.batch_size)
hidden_size = int(args.hidden_size)
latent_size = int(args.latent_size)
depth = int(args.depth)
beta = float(args.beta)
lr = float(args.lr)

MAX_EPOCH = 100
PRINT_ITER = 20

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    dataset.training = True
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=JTNNCollator(dataset.vocab, True),
        drop_last=True,
        worker_init_fn=worker_init_fn)

    model = DGLJTNNVAE(vocab_file=vocab_file,
                       hidden_size=hidden_size,
                       latent_size=latent_size,
                       depth=depth)

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)
    model = model.to(device)
    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)

    for epoch in range(MAX_EPOCH):
        word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0

        for it, batch in enumerate(dataloader):
            batch['mol_graph_batch'] = batch['mol_graph_batch'].to(device)
            model.zero_grad()
            try:
                loss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta)
            except:
                print([t.smiles for t in batch['mol_trees']])
                raise
            loss.backward()
            optimizer.step()

            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc

            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                steo_acc = steo_acc / PRINT_ITER * 100

                print("KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Loss: %.6f" % (
                    kl_div, word_acc, topo_acc, assm_acc, steo_acc, loss.item()))
                word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0
                sys.stdout.flush()

            if (it + 1) % 1500 == 0:  # Fast annealing
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])
                torch.save(model.state_dict(),
                           args.save_path + "/model.iter-%d-%d" % (epoch, it + 1))

        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        torch.save(model.state_dict(), args.save_path + "/model.iter-" + str(epoch))

if __name__ == '__main__':
    train()
