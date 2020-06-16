# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from logger import Logger

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import SAGEConv

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=F.relu))
        # hidden layers
        for i in range(num_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(n_hidden, out_feats, activation=None))
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, x):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(g, x)
        return x


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 num_layers,
                 dropout,
                 activation=F.relu):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, "mean", feat_drop=0., activation=activation))
        # hidden layers
        for i in range(num_layers - 2):
            self.layers.append(SAGEConv(n_hidden, n_hidden, "mean", feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, out_feats, "mean", feat_drop=dropout, activation=None)) # activation None

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, x):
        for layer in self.layers:
            x = layer(g, x)
        return x


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for layer in self.lins:
            layer.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, g, x, splitted_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = splitted_edge['train']['edge'].to(x.device)

    total_loss = total_samples = 0
    for perm in DataLoader(
            range(pos_train_edge.size(0)), batch_size, shuffle=True):

        h = model(g, x)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(
            0, x.size(0), edge.size(), dtype=torch.long, device=x.device)

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_samples = pos_out.size(0)
        total_loss += loss.item() * num_samples
        total_samples += num_samples

    return total_loss / total_samples


@torch.no_grad()
def test(model, predictor, g, x, splitted_edge, evaluator, batch_size):
    model.eval()

    h = model(g, x)

    pos_train_edge = splitted_edge['train']['edge'].to(x.device)
    pos_valid_edge = splitted_edge['valid']['edge'].to(x.device)
    neg_valid_edge = splitted_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = splitted_edge['test']['edge'].to(x.device)
    neg_test_edge = splitted_edge['test']['edge_neg'].to(x.device)

    # Positive training edges
    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size=batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_preds = torch.cat(pos_train_preds, dim=0)

    # Positive validation edges
    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size=batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_preds = torch.cat(pos_valid_preds, dim=0)

    # Negative validation edges
    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size=batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_preds = torch.cat(neg_valid_preds, dim=0)

    # Positive test edges
    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size=batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_preds = torch.cat(pos_test_preds, dim=0)

    # Negative test edges
    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size=batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_preds = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_preds,
            'y_pred_neg': neg_valid_preds
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_preds,
            'y_pred_neg': neg_valid_preds
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_preds,
            'y_pred_neg': neg_test_preds
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-PPA (Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglLinkPropPredDataset(name='ogbl-ppa')
    # Get DGLGraph
    data = dataset[0]
    data.readonly(False)
    data.add_edges(data.nodes(), data.nodes())
    splitted_edge = dataset.get_edge_split()

    if args.use_node_embedding:
        # Todo: prepare node embeddings using node2vec
        x = data.ndata['feat'].float()
        x = torch.cat([x, torch.load('embedding.pt')], dim=-1)
        x = x.to(device)
    else:
        x = data.ndata['feat'].float().to(device)

    if args.use_sage:
        model = SAGE(
            x.size(-1), args.hidden_channels, args.hidden_channels,
            args.num_layers, args.dropout).to(device)
    else:
        model = GCN(
            x.size(-1), args.hidden_channels, args.hidden_channels,
            args.num_layers, args.dropout).to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ppa')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, x, splitted_edge, optimizer,
                         args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, x, splitted_edge,
                               evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
