# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import torch
import torch.nn.functional as F

from dgllife.model import GCN, GraphSAGE, HadamardLinkPredictor
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.utils.data import DataLoader

from logger import Logger

def train(model, predictor, g, x, splitted_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = splitted_edge['train']['edge'].to(x.device)

    total_loss = total_samples = 0
    for perm in DataLoader(
            range(pos_train_edge.size(0)), batch_size, shuffle=True):

        h = model(g, x)

        edge = pos_train_edge[perm].t()
        pos_out = torch.sigmoid(predictor(h[edge[0]], h[edge[1]]))
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(
            0, x.size(0), edge.size(), dtype=torch.long, device=x.device)

        neg_out = torch.sigmoid(predictor(h[edge[0]], h[edge[1]]))
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
        pos_train_preds += [torch.sigmoid(
            predictor(h[edge[0]], h[edge[1]])).squeeze().cpu()]
    pos_train_preds = torch.cat(pos_train_preds, dim=0)

    # Positive validation edges
    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size=batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [torch.sigmoid(
            predictor(h[edge[0]], h[edge[1]])).squeeze().cpu()]
    pos_valid_preds = torch.cat(pos_valid_preds, dim=0)

    # Negative validation edges
    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size=batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [torch.sigmoid(
            predictor(h[edge[0]], h[edge[1]])).squeeze().cpu()]
    neg_valid_preds = torch.cat(neg_valid_preds, dim=0)

    # Positive test edges
    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size=batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [torch.sigmoid(
            predictor(h[edge[0]], h[edge[1]])).squeeze().cpu()]
    pos_test_preds = torch.cat(pos_test_preds, dim=0)

    # Negative test edges
    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size=batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [torch.sigmoid(
            predictor(h[edge[0]], h[edge[1]])).squeeze().cpu()]
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
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use gpu for computation (default: False)')
    parser.add_argument('--log_steps', type=int, default=1,
                        help='Print training progress every {log_steps} epochs (default: 1)')
    parser.add_argument('--use_sage', action='store_true',
                        help='Use GraphSAGE rather than GCN (default: False)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers to use as well as '
                             'linear layers to use for final link prediction (default: 3)')
    parser.add_argument('--hidden_feats', type=int, default=256,
                        help='Size for hidden representations (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout (default: 0.0)')
    parser.add_argument('--batch_size', type=int, default=64 * 1024,
                        help='Batch size to use for link prediction (default: 64 * 1024)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for training (default: 20)')
    parser.add_argument('--eval_steps', type=int, default=1,
                        help='Evaluate hits@100 every {eval_steps} epochs (default: 1)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of random experiments to perform (default: 10)')
    args = parser.parse_args()
    print(args)

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    dataset = DglLinkPropPredDataset(name='ogbl-ppa')
    # Get DGLGraph
    data = dataset[0]
    data.readonly(False)
    data.add_edges(data.nodes(), data.nodes())
    splitted_edge = dataset.get_edge_split()
    x = data.ndata['feat'].float().to(device)

    if args.use_sage:
        model = GraphSAGE(in_feats=x.size(-1),
                          hidden_feats=[args.hidden_feats for _ in range(args.num_layers)],
                          activation=[F.relu for _ in range(args.num_layers - 1)] + [None],
                          dropout=[0] + [args.dropout for _ in range(args.num_layers - 1)]).to(device)
    else:
        model = GCN(in_feats=x.size(-1),
                    hidden_feats=[args.hidden_feats for _ in range(args.num_layers)],
                    activation=[F.relu for _ in range(args.num_layers - 1)] + [None],
                    residual=[False for _ in range(args.num_layers)],
                    batchnorm=[False for _ in range(args.num_layers)],
                    dropout=[args.dropout for _ in range(args.num_layers - 1)] + [0]).to(device)

    predictor = HadamardLinkPredictor(in_feats=args.hidden_feats,
                                      hidden_feats=args.hidden_feats,
                                      num_layers=args.num_layers,
                                      n_tasks=1,
                                      dropout=args.dropout).to(device)

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
