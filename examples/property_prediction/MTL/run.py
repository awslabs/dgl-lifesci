import numpy as np
import time
import torch
import torch.nn as nn

from dgllife.utils import EarlyStopping, Meter
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import get_label_mean_and_std, collate, load_model

def regress(args, model, bg):
    bg = bg.to(args['device'])
    node_feats = bg.ndata.pop('hv')
    edge_feats = bg.edata.pop('he')
    return model(bg, node_feats, edge_feats)

def run_a_train_epoch(args, model, data_loader, criterion, optimizer):
    model.train()
    train_meter = Meter(args['train_mean'], args['train_std'])
    epoch_loss = torch.zeros(len(args['tasks']))
    for _, batch_data in enumerate(data_loader):
        _, bg, labels, masks = batch_data
        labels, masks = labels.to(args['device']), masks.to(args['device'])
        prediction = regress(args, model, bg)
        # Normalize the labels so that the scale of labels will be similar
        loss = criterion(prediction, (labels - args['train_mean']) / args['train_std'])
        # Mask non-existing labels
        loss = (loss * (masks != 0).float()).sum(0)
        # Update epoch loss
        epoch_loss = epoch_loss + loss.detach().cpu().data
        # Average the loss over batch
        loss = loss.sum() / bg.batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
    epoch_loss = epoch_loss / len(data_loader.dataset)
    epoch_loss = epoch_loss.cpu().detach().tolist()

    return epoch_loss, train_meter.pearson_r2(), train_meter.mae()

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter(args['train_mean'], args['train_std'])
    with torch.no_grad():
        for _, batch_data in enumerate(data_loader):
            _, bg, labels, masks = batch_data
            prediction = regress(args, model, bg)
            eval_meter.update(prediction, labels, masks)

    return eval_meter.pearson_r2(), eval_meter.mae()

def log_model_evaluation(args, model, train_loader, val_loader, test_loader):
    with open(args['result_path'] + '/results.txt', 'w') as f:
        def _log_values(metric, train_values, val_values, test_values):
            f.write('{}\n'.format(metric))
            headline = '|            | averaged |'
            for t in args['tasks']:
                headline += ' {:15} |'.format(t)
            headline += '\n'
            f.write(headline)
            f.write('| ' + '-' * (len(headline) - 5) + ' |\n')
            for name, values in {'Training': train_values,
                                 'Validation': val_values,
                                 'Test': test_values}.items():
                row = '| {:10s} | {:8.3f} |'.format(name, np.mean(values))
                for t in range(len(args['tasks'])):
                    row += ' {:15.3f} |'.format(values[t])
                row += '\n'
                f.write(row)
            f.write('| ' + '-' * (len(headline) - 5) + ' |\n')
            f.write('\n')

        train_r2, train_mae = run_an_eval_epoch(args, model, train_loader)
        val_r2, val_mae = run_an_eval_epoch(args, model, val_loader)
        test_r2, test_mae = run_an_eval_epoch(args, model, test_loader)
        _log_values('r2', train_r2, val_r2, test_r2)
        _log_values('mae', train_mae, val_mae, test_mae)

def main(args, node_featurizer, edge_featurizer, train_set, val_set, test_set):
    # Record starting time
    t0 = time.time()

    train_mean, train_std = get_label_mean_and_std(train_set)
    train_mean, train_std = train_mean.to(args['device']), train_std.to(args['device'])
    args['train_mean'], args['train_std'] = train_mean, train_std

    train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'],
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'],
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                             shuffle=False, collate_fn=collate)

    model = load_model(args, node_featurizer, edge_featurizer).to(args['device'])
    criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    stopper = EarlyStopping(patience=args['patience'],
                            filename=args['result_path'] + '/model.pth')

    for epoch in range(args['num_epochs']):
        loss, train_r2, train_mae = run_a_train_epoch(
            args, model, train_loader, criterion, optimizer)
        print('Epoch {:d}/{:d} | training | averaged loss {:.4f} | '
              'averaged r2 {:.4f} | averaged mae {:.4f}'.format(
            epoch + 1, args['num_epochs'], float(np.mean(loss)),
            float(np.mean(train_r2)), float(np.mean(train_mae))))

        # Validation and early stop
        val_r2, val_mae = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(float(np.mean(val_r2)), model)
        print('Epoch {:d}/{:d} | validation | current r2 {:.4f} | best r2 {:.4f} | mae {:.4f}'.format(
            epoch + 1, args['num_epochs'], float(np.mean(val_r2)),
            stopper.best_score, float(np.mean(val_mae))))

        if early_stop:
            break

    print('It took {:.4f}s to complete the task'.format(time.time() - t0))
    stopper.load_checkpoint(model)
    log_model_evaluation(args, model, train_loader, val_loader, test_loader)
