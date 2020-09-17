# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from dgllife.data import JTVAEDataset, JTVAECollator
from dgllife.model import DGLJTNNVAE, load_pretrained
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import worker_init_fn, get_vocab_file

@torch.no_grad()
def main(args):
    worker_init_fn(None)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if args.model_path is not None:
        model = DGLJTNNVAE(vocab_file=get_vocab_file(args.vocab),
                           hidden_size=args.hidden_size,
                           latent_size=args.latent_size,
                           depth=args.depth)
        model.load_state_dict(torch.load(args.model_path))
    else:
        model = load_pretrained("JTNN_ZINC")
    print("# model parameters: {:d}K".format(
        sum([x.nelement() for x in model.parameters()]) // 1000))

    dataset = JTVAEDataset(data=args.data, vocab=model.vocab, training=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=JTVAECollator(False),
        worker_init_fn=worker_init_fn)

    acc = 0.0
    tot = 0
    model = model.to(device)
    model.eval()
    for it, batch in enumerate(tqdm(dataloader)):
        gt_smiles = batch['mol_trees'][0].smiles
        batch = dataset.move_to_device(batch, device)
        try:
            _, tree_vec, mol_vec = model.encode(batch)

            tree_mean = model.T_mean(tree_vec)
            # Following Mueller et al.
            tree_log_var = -torch.abs(model.T_var(tree_vec))
            epsilon = torch.randn(1, model.latent_size // 2).to(device)
            tree_vec = tree_mean + torch.exp(tree_log_var // 2) * epsilon

            mol_mean = model.G_mean(mol_vec)
            # Following Mueller et al.
            mol_log_var = -torch.abs(model.G_var(mol_vec))
            epsilon = torch.randn(1, model.latent_size // 2).to(device)
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

    print("Reconstruction Accuracy: {}".format(acc / tot))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Evaluation for JTVAE")

    parser.add_argument("-da", "--data", default='test',
                        help="Data for evaluation. 1) If 'train' or 'test', it will perform the "
                             "evaluation on the training or test subset of ZINC. 2) Otherwise, it "
                             "should be the path to a .txt file of compounds, with one SMILES per line.")
    parser.add_argument("-v", "--vocab", default='zinc',
                        help="Vocabulary to use. 1) If 'zinc' or 'guacamol', it will use the "
                             "vocabulary extracted from ZINC or GuacaMol. 2) Otherwise, it "
                             "should be the path to a file of vocabulary generated with vocab.py")
    parser.add_argument("-m", "--model-path", default=None,
                        help="Path to a pre-trained model. If not specified, it will load a model "
                             "pre-trained on ZINC.")
    parser.add_argument("-w", "--hidden-size", type=int, default=450,
                        help="Hidden size of representation vector, "
                             "should be consistent with pre-trained model")
    parser.add_argument("-l", "--latent-size", type=int, default=56,
                        help="Latent size of node features and edge features, "
                             "should be consistent with pre-trained model")
    parser.add_argument("-d", "--depth", type=int, default=3,
                        help="Depth of message passing hops, "
                             "should be consistent with pre-trained model")
    args = parser.parse_args()
    main(args)
