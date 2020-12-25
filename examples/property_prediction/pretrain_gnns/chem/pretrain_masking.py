import argparse
import tqdm
from functools import partial
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import random

import dgl
from dgllife.utils import PretrainAtomFeaturizer
from dgllife.utils import PretrainBondFeaturizer
from dgllife.utils import smiles_to_bigraph

from model import GINMaskingModel
from utils import *


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def collate_masking(graphs, args):
    """
    Collate function used for pretrain_masking task to randomly mask nodes/edges
    """
    bg = dgl.batch(graphs)
    # masked nodes/edges indices in batched graph
    masked_nodes_indices = torch.LongTensor(
        random.sample(range(bg.num_nodes()), int(bg.num_nodes() * args.mask_rate)))
    if args.mask_edge:
        masked_edges_indices = mask_edges(bg, masked_nodes_indices)

    node_feats = [
        bg.ndata.pop('atomic_number'),
        bg.ndata.pop('chirality_type')
    ]
    edge_feats = [
        bg.edata.pop('bond_type'),
        bg.edata.pop('bond_direction_type')
    ]
    # export masked nodes labels
    masked_nodes_labels = node_feats[0][masked_nodes_indices]
    # mask these nodes labels
    node_feats[0][masked_nodes_indices] = 118

    if args.mask_edge:
        # export masked edges labels
        masked_edges_labels = edge_feats[0][masked_edges_indices]
        # mask these edges labels
        edge_feats[0][masked_edges_indices] = 5
    else:
        # if no edge masking
        masked_edges_indices = 0
        masked_edges_labels = 0

    # return batched graph, node features, edge features,
    # masked node indices, masked edge indices, masked nodes labels, masked edges labels
    return bg, node_feats, edge_feats, \
           (masked_nodes_indices, masked_edges_indices), (masked_nodes_labels, masked_edges_labels)


def train(args, model_list, train_dataloader, optimizer, criterion, device):
    if args.mask_edge:
        model, node_linear, edge_linear = model_list
    else:
        model, node_linear = model_list

    for epoch in range(1, args.epochs):
        model.train()
        node_linear.train()
        if args.mask_edge:
            edge_linear.train()
        with tqdm.tqdm(train_dataloader) as tq_train:
            for step, (bg, node_feats, edge_feats, masked_indices, masked_labels) in enumerate(tq_train):
                masked_nodes_indices, masked_edges_indices = masked_indices
                masked_nodes_labels, masked_edges_labels = masked_labels

                node_feats = [n.to(device) for n in node_feats]
                edge_feats = [e.to(device) for e in edge_feats]
                masked_nodes_indices = masked_nodes_indices.to(device)
                masked_nodes_labels = masked_nodes_labels.to(device)
                if args.mask_edge:
                    masked_edges_indices = masked_edges_indices.to(device)
                    masked_edges_labels = masked_edges_labels.to(device)
                bg = bg.to(device)

                logits = model(bg, node_feats, edge_feats)

                pred_node = node_linear(logits)
                loss_node = criterion(pred_node[masked_nodes_indices], masked_nodes_labels)
                node_acc = compute_accuracy(pred_node[masked_nodes_indices], masked_nodes_labels)
                if args.mask_edge:
                    pred_edge = edge_linear(logits)
                    # for every edge, add two corresponding node feature.
                    masked_edges_logits = pred_edge[bg.find_edges(masked_edges_indices)[0]] + \
                                          pred_edge[bg.find_edges(masked_edges_indices)[1]]
                    loss_edge = criterion(masked_edges_logits, masked_edges_labels)
                    edge_acc = compute_accuracy(masked_edges_logits, masked_edges_labels)
                    loss = loss_node + loss_edge
                else:
                    loss = loss_node
                    edge_acc = 0

                tq_train.set_postfix_str(
                    "Epoch: {} Step: {} Loss: {:.4f} node_acc: {:.4f} edge_acc: {:.4f}".format(epoch,
                                                                                               step + 1,
                                                                                               loss.item(),
                                                                                               node_acc,
                                                                                               edge_acc), refresh=False)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def main():
    parser = argparse.ArgumentParser(description='pretrain_masking')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='ratio of masking nodes (atoms). (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', action='store_true',
                        help='whether to mask all edges connected to masked nodes (atoms) (default: False)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat. (default: last)')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset for pretraining. (default: zinc_standard_agent)')
    parser.add_argument('--output_model_file', type=str, default='model.pth',
                        help='filename to output the model. (default: model.pth)')
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataset loading. (default: 8)')
    args = parser.parse_args()
    print(args)

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dgl.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.dataset != 'zinc_standard_agent':
        raise ValueError('Dataset should be zinc_standard_agent.')
    df = pd.read_csv('./zinc.csv')

    atom_featurizer = PretrainAtomFeaturizer()
    bond_featurizer = PretrainBondFeaturizer()
    dataset = PretrainMaskingMoleculeCSVDataset(df=df,
                                                smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                                node_featurizer=atom_featurizer,
                                                edge_featurizer=bond_featurizer,
                                                smiles_column='smiles')

    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  collate_fn=partial(collate_masking, args=args))

    # 118 atom number plus 1 masked node type = 119
    # 4 bond type plus self-loop plus 1 masked node type = 6
    model = GINMaskingModel(num_node_emb_list=[119, 4],
                            num_edge_emb_list=[6, 3],
                            num_layers=args.num_layer,
                            emb_dim=args.emb_dim,
                            JK=args.JK,
                            dropout=args.dropout_ratio)
    node_linear = nn.Linear(args.emb_dim, 119)
    if args.mask_edge:
        edge_linear = nn.Linear(args.emb_dim, 6)

    model.to(device)
    node_linear.to(device)
    if args.mask_edge:
        edge_linear.to(device)

    if args.mask_edge:
        all_params = itertools.chain(model.parameters(), node_linear.parameters(), edge_linear.parameters())
    else:
        all_params = itertools.chain(model.parameters(), node_linear.parameters())

    optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss()

    if args.mask_edge:
        model_list = (model, node_linear, edge_linear)
    else:
        model_list = (model, node_linear)
    train(args, model_list, train_dataloader, optimizer, criterion, device)

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file)


if __name__ == "__main__":
    main()
