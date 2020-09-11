# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import rdkit
import torch

from dgllife.model import DGLJTNNVAE, load_pretrained
from torch.utils.data import DataLoader
from tqdm import tqdm

from datautils import JTNNCollator, JTNNDataset

def worker_init_fn(id_):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

worker_init_fn(None)

parser = argparse.ArgumentParser(description="Evaluation for JTNN",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--train", dest="train",
                    default='test', help='Training file name')
parser.add_argument("-v", "--vocab", dest="vocab",
                    default='zinc', help='Vocab file name')
parser.add_argument("-m", "--model", dest="model_path", default=None,
                    help="Pre-trained model to be loaded for evalutaion. If not specified,"
                         " would use pre-trained model from model zoo")
parser.add_argument("-w", "--hidden", dest="hidden_size", default=450,
                    help="Hidden size of representation vector, "
                         "should be consistent with pre-trained model")
parser.add_argument("-l", "--latent", dest="latent_size", default=56,
                    help="Latent Size of node(atom) features and edge(atom) features, "
                         "should be consistent with pre-trained model")
parser.add_argument("-d", "--depth", dest="depth", default=3,
                    help="Depth of message passing hops, "
                         "should be consistent with pre-trained model")
args = parser.parse_args()

dataset = JTNNDataset(data=args.train, vocab=args.vocab, training=False)
vocab_file = dataset.vocab_file

hidden_size = int(args.hidden_size)
latent_size = int(args.latent_size)
depth = int(args.depth)

MAX_EPOCH = 100
PRINT_ITER = 20

@torch.no_grad()
def reconstruct():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    dataset.training = False
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=JTNNCollator(dataset.vocab, False),
        drop_last=True,
        worker_init_fn=worker_init_fn)

    model = DGLJTNNVAE(vocab_file=vocab_file,
                       hidden_size=hidden_size,
                       latent_size=latent_size,
                       depth=depth)

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    else:
        model = load_pretrained("JTNN_ZINC")

    print("Model #Params: %dK" %
          (sum([x.nelement() for x in model.parameters()]) / 1000,))

    # Just an example of molecule decoding; in reality you may want to sample
    # tree and molecule vectors.
    acc = 0.0
    tot = 0
    model = model.to(device)
    model.eval()
    for it, batch in enumerate(tqdm(dataloader)):
        gt_smiles = batch['mol_trees'][0].smiles
        # print(gt_smiles)
        batch = model.move_to_device(batch, device)
        try:
            _, tree_vec, mol_vec = model.encode(batch)

            tree_mean = model.T_mean(tree_vec)
            # Following Mueller et al.
            tree_log_var = -torch.abs(model.T_var(tree_vec))
            mol_mean = model.G_mean(mol_vec)
            # Following Mueller et al.
            mol_log_var = -torch.abs(model.G_var(mol_vec))

            epsilon = torch.randn(1, model.latent_size // 2).cuda()
            tree_vec = tree_mean + torch.exp(tree_log_var // 2) * epsilon
            epsilon = torch.randn(1, model.latent_size // 2).cuda()
            mol_vec = mol_mean + torch.exp(mol_log_var // 2) * epsilon
            dec_smiles = model.decode(tree_vec, mol_vec)

            if dec_smiles == gt_smiles:
                acc += 1
            tot += 1
        except Exception as e:
            print("Failed to encode: {}".format(gt_smiles))
            print(e)

        if it % 20 == 1:
            print("Progress {}/{}; Current Reconstruction Accuracy: {:.4f}".format(
                it, len(dataloader), acc / tot))
    return acc / tot

if __name__ == '__main__':
    reconstruct_acc = reconstruct()
    print("Reconstruction Accuracy: {}".format(reconstruct_acc))
