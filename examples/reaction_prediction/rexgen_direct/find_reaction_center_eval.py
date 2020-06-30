# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from dgllife.data import USPTOCenter, WLNCenterDataset
from dgllife.model import WLNReactionCenter, load_pretrained
from torch.utils.data import DataLoader

from utils import reaction_center_final_eval, set_seed, collate_center, mkdir_p

def main(args):
    set_seed()
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')
    # Set current device
    torch.cuda.set_device(args['device'])

    if args['test_path'] is None:
        test_set = USPTOCenter('test', num_processes=args['num_processes'], load=args['load'])
    else:
        test_set = WLNCenterDataset(raw_file_path=args['test_path'],
                                    mol_graph_path=args['test_path'] + '.bin',
                                    num_processes=args['num_processes'],
                                    load=args['load'],
                                    reaction_validity_result_prefix='test')
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             collate_fn=collate_center, shuffle=False)

    if args['model_path'] is None:
        model = load_pretrained('wln_center_uspto')
    else:
        model = WLNReactionCenter(node_in_feats=args['node_in_feats'],
                                  edge_in_feats=args['edge_in_feats'],
                                  node_pair_in_feats=args['node_pair_in_feats'],
                                  node_out_feats=args['node_out_feats'],
                                  n_layers=args['n_layers'],
                                  n_tasks=args['n_tasks'])
        model.load_state_dict(torch.load(
            args['model_path'], map_location='cpu')['model_state_dict'])
    model = model.to(args['device'])

    print('Evaluation on the test set.')
    test_result = reaction_center_final_eval(
        args, args['top_ks_test'], model, test_loader, args['easy'])
    print(test_result)
    with open(args['result_path'] + '/test_eval.txt', 'w') as f:
        f.write(test_result)

if __name__ == '__main__':
    from argparse import ArgumentParser

    from configure import reaction_center_config

    parser = ArgumentParser(description='Reaction Center Identification -- Evaluation')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model. If None, we will directly evaluate '
                             'a pretrained model on the test set.')
    parser.add_argument('--result-path', type=str, default='center_results',
                        help='Path where we saved model training and evaluation results')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Path to a new test set.'
                             'If None, we will use the default test set in USPTO.')
    parser.add_argument('--easy', action='store_true', default=False,
                        help='Whether to exclude reactants not contributing heavy atoms to the '
                             'product in top-k atom pair selection, which will make the '
                             'task easier.')
    parser.add_argument('-np', '--num-processes', type=int, default=32,
                        help='Number of processes to use for data pre-processing')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Whether to load constructed DGLGraphs. This is desired when the '
                             'evaluation is performed multiple times and the dataset is large.')
    args = parser.parse_args().__dict__
    args.update(reaction_center_config)

    assert args['max_k'] >= max(args['top_ks_test']), \
        'Expect max_k to be no smaller than the possible options ' \
        'of top_ks_test, got {:d} and {:d}'.format(args['max_k'], max(args['top_ks_test']))
    mkdir_p(args['result_path'])
    main(args)
