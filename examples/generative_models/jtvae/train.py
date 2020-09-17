# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from dgllife.data import JTVAEDataset, JTVAECollator
from dgllife.model import DGLJTNNVAE
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import worker_init_fn, get_vocab_file

def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')

    worker_init_fn(None)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = DGLJTNNVAE(vocab_file=get_vocab_file(args.vocab),
                       hidden_size=args.hidden_size,
                       latent_size=args.latent_size,
                       depth=args.depth)
    print("# model parameters: {:d}K".format(
        sum([x.nelement() for x in model.parameters()]) // 1000))

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    dataset = JTVAEDataset(data=args.data, vocab=model.vocab, training=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=JTVAECollator(True),
        drop_last=True,
        worker_init_fn=worker_init_fn)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)

    model = model.to(device)
    model.train()
    for epoch in range(args.max_epoch):
        word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0

        for it, batch in enumerate(tqdm(dataloader)):
            batch = dataset.move_to_device(batch, device)
            model.zero_grad()
            try:
                loss, kl_div, wacc, tacc, sacc, dacc = model(batch, args.beta)
            except:
                print([t.smiles for t in batch['mol_trees']])
                raise
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
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Training for JTVAE")
    parser.add_argument("-da", "--data", default='train',
                        help="Data for training. 1) If 'train', it will perform the training on the "
                             "training subset of ZINC. 2) Otherwise, it should be the path to a .txt "
                             "file of compounds, with one SMILES per line.")
    parser.add_argument("-v", "--vocab", default='zinc',
                        help="Vocabulary to use. 1) If 'zinc' or 'guacamol', it will use the "
                             "vocabulary extracted from ZINC or GuacaMol. 2) Otherwise, it "
                             "should be the path to a file of vocabulary generated with vocab.py")
    parser.add_argument("-s", "--save_dir", dest="save_path", default='./',
                        help="Path to save checkpoint models, default to be current working directory")
    parser.add_argument("-m", "--model", dest="model_path", default=None,
                        help="Path to load pre-trained model")
    parser.add_argument("-b", "--batch-size", type=int, default=40,
                        help="Batch size")
    parser.add_argument("-w", "--hidden-size", type=int, default=200,
                        help="Size of representation vectors")
    parser.add_argument("-l", "--latent-size", type=int, default=56,
                        help="Latent Size of node(atom) features and edge(atom) features")
    parser.add_argument("-d", "--depth", type=int, default=3,
                        help="Depth of message passing hops")
    parser.add_argument("-z", "--beta", type=float, default=1.0,
                        help="Coefficient of KL Divergence term")
    parser.add_argument("-q", "--lr", type=float, default=1e-3,
                        help="Learning Rate")
    parser.add_argument("-me", "--max-epoch", type=int, default=100,
                        help='Max number of epochs for training')
    parser.add_argument("-pi", "--print-iter", type=int, default=20,
                        help='Frequency of printing messages')
    args = parser.parse_args()

    main(args)
