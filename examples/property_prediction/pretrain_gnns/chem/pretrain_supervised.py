# -*- coding: utf-8 -*-
import argparse
import pickle
import tqdm
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dgl
from dgllife.utils import PretrainAtomFeaturizer
from dgllife.utils import PretrainBondFeaturizer
from dgllife.utils import smiles_to_bigraph
from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive
from dgllife.model.model_zoo.gin_predictor import GINPredictor

from utils import PretrainDataset


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.stack(labels)
    return batched_graph, batched_labels


def train(args, model, train_dataloader, optimizer, criterion, device):
    for epoch in range(1, args.epochs):
        model.train()
        with tqdm.tqdm(train_dataloader) as tq_train:
            for step, (bg, labels) in enumerate(tq_train):
                node_feats = [
                    bg.ndata.pop('atomic_number'),
                    bg.ndata.pop('chirality_type')
                ]
                edge_feats = [
                    bg.edata.pop('bond_type'),
                    bg.edata.pop('bond_direction_type')
                ]
                bg = bg.to(device)
                node_feats = [n.to(device) for n in node_feats]
                edge_feats = [e.to(device) for e in edge_feats]

                logits = model(bg, node_feats, edge_feats)

                # The possible values are {-1, 0, 1} in `labels`.
                # "1" means positive, and "-1" means negative. "0" indicates the molecule is invalid.
                is_valid = labels ** 2 > 0
                is_valid = is_valid.to(device)
                labels = labels.type_as(logits)
                # Negative labels will be changed from "-1" to "0"
                loss = criterion(logits, (labels + 1) / 2)
                # Replace invalid molecule labels from "-1" to "0".
                loss = torch.where(is_valid, loss, torch.zeros(
                    loss.shape).to(loss.device).to(loss.dtype))
                loss = torch.sum(loss) / torch.sum(is_valid)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tq_train.set_postfix_str(
                    "Epoch: {} Step: {} Loss: {:.4f}".format(epoch,
                                                             step + 1,
                                                             loss.item()), refresh=False)


def main():
    parser = argparse.ArgumentParser(description='pretrain_supervised')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any. (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training. (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train. (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate. (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay. (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers. (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions. (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio. (default: 0.2)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling, or readout.'
                             'for computing graph representations out of node representations, '
                             'which can be `sum`, `mean`, `max` or `attention`. (default: mean)'
                             '`sum`: apply sum pooling over the nodes in the graph.'
                             '`mean`: apply average pooling over the nodes in the graph.'
                             '`max`: apply max pooling over the nodes in the graph.'
                             '`attention`: apply Global Attention Pooling over the nodes in the graph.')  # TODO: set2set
    parser.add_argument('--JK', type=str, default="last",
                        help='JK for jumping knowledge '
                             'decides how we are going to combine the all-layer node representations for the final output.'
                             'It can be `concat`, `last`, `max` or `sum`. (default: last)'
                             '`concat`: concatenate the output node representations from all GIN layers'
                             '`last`: use the node representations from the last GIN layer'
                             '`max`: apply max pooling to the node representations across all GIN layers'
                             '`sum`: sum the output node representations from all GIN layers')
    parser.add_argument('--dataset', type=str, default='chembl_filtered',
                        help='path of the dataset. For now, only classification (default: chembl_filtered)')
    parser.add_argument('--input_model_file', type=str, default=None,
                        help='filename to read the model if there is any. (default: no model to load.)')
    parser.add_argument('--output_model_file', type=str, default='pretrain_supervised.pth',
                        help='filename to output the pre-trained model. (default: pretrain_supervised.pth)')
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataset loading. (default: 8)')
    args = parser.parse_args()
    print(args)

    # set seed
    torch.manual_seed(args.seed)
    dgl.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")

    model = GINPredictor(num_node_emb_list=[119, 4],
                         num_edge_emb_list=[6, 3],
                         num_layers=args.num_layer,
                         emb_dim=args.emb_dim,
                         JK=args.JK,
                         dropout=args.dropout_ratio,
                         readout=args.graph_pooling,
                         n_tasks=1310)

    if args.input_model_file is not None:
        model.gnn.load_state_dict(torch.load(args.input_model_file))
    model.to(device)

    if args.dataset == 'chembl_filtered':
        url = 'dataset/pretrain_gnns.zip'
        data_path = get_download_dir() + '/pretrain_gnns.zip'
        dir_path = get_download_dir() + '/pretrain_gnns'
        download(_get_dgl_url(url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        with open(dir_path + '/supervised_chembl_rev.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        with open(args.dataset, 'rb') as f:
            data = pickle.load(f)

    atom_featurizer = PretrainAtomFeaturizer()
    bond_featurizer = PretrainBondFeaturizer()
    dataset = PretrainDataset(data=data,
                              smiles_to_graph=partial(
                                  smiles_to_bigraph, add_self_loop=True),
                              node_featurizer=atom_featurizer,
                              edge_featurizer=bond_featurizer,
                              task='supervised')

    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    train(args, model, train_dataloader, optimizer, criterion, device)

    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), args.output_model_file)


if __name__ == "__main__":
    main()
