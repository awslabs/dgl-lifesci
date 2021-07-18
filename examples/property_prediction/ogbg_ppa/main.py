import argparse
import numpy as np
import time
import torch
import torch.optim as optim

from dgllife.model import GNNOGBPredictor
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator, collate_dgl
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model, device, loader, criterion, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        bg, labels = batch
        bg, labels = bg.to(device), labels.to(device)
        nfeats = bg.ndata['h']
        efeats = bg.edata['feat']
        # only one node
        if bg.batch_size == 1:
            pass
        else:
            pred = model(bg, nfeats, efeats)
            optimizer.zero_grad()
            loss = criterion(pred.to(torch.float32), labels.view(-1,))
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        bg, labels = batch
        bg, labels = bg.to(device), labels.to(device)
        nfeats = bg.ndata['h']
        efeats = bg.edata['feat']
        # only one node
        if bg.batch_size == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(bg, nfeats, efeats)

            y_true.append(labels.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa with DGL')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gcn, gin-virtual, gcn-virtual (default: gin-virtual)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--hidden_feats', type=int, default=300,
                        help='number of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs for training (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa)')
    parser.add_argument('--filename', type=str,
                        help='filename to output result')
    args = parser.parse_args()

    if args.filename is None:
        args.filename = args.gnn

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # data loading and splitting
    dataset = DglGraphPropPredDataset(name=args.dataset)
    # initialize node features
    for i in range(len(dataset)):
        dataset[i][0].ndata['h'] = torch.zeros(dataset[i][0].num_nodes()).long()
    splitted_idx = dataset.get_idx_split()

    # automatic evaluator taking dataset name as input
    evaluator = Evaluator(args.dataset)

    # using collate_dgl
    train_loader = DataLoader(dataset[splitted_idx["train"]], batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_dgl, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[splitted_idx["valid"]], batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_dgl, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[splitted_idx["test"]], batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_dgl, num_workers=args.num_workers)

    if args.gnn == 'gin':
        gnn_type = 'gin'
        virtual_node = False
    if args.gnn == 'gcn':
        gnn_type = 'gcn'
        virtual_node = False
    if args.gnn == 'gin-virtual':
        gnn_type = 'gin'
        virtual_node = True
    if args.gnn == 'gcn-virtual':
        gnn_type = 'gcn'
        virtual_node = True

    model = GNNOGBPredictor(in_edge_feats=dataset[0][0].edata['feat'].shape[-1],
                            hidden_feats=args.hidden_feats,
                            n_layers=args.n_layers,
                            n_tasks=int(dataset.num_classes),
                            dropout=args.dropout,
                            gnn_type=gnn_type,
                            virtual_node=virtual_node).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []
    time_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        t0 = time.time()
        train(model, device, train_loader, criterion, optimizer)
        t1 = time.time()
        if epoch >= 3:
            time_curve.append(t1 - t0)
        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        if epoch >= 3:
            print('Training Time: ', time_curve[-1])

        train_curve.append(train_perf['acc'])
        valid_curve.append(valid_perf['acc'])
        test_curve.append(test_perf['acc'])

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    print('Avg Training Time: ', np.mean(time_curve))
    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
