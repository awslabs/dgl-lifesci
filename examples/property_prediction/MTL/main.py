if __name__ == '__main__':
    from argparse import ArgumentParser

    from configure import configs

    parser = ArgumentParser('(Multitask) Molecular Property Prediction with GNNs for a user-specified .csv file.')
    parser.add_argument('-c', '--csv-path', type=str, required=True,
                        help='Path to a csv file for loading a dataset.')
    parser.add_argument('-m', '--model', type=str,
                        choices=['GCN', 'GAT', 'MPNN', 'AttentiveFP'],
                        help='Model to use')
    parser.add_argument('--mode', type=str, choices=['parallel', 'bypass'],
                        help='Architecture to use for multitask learning')
    parser.add_argument('-n', '--num-epochs', type=int, default=4000,
                        help='Maximum number of epochs allowed for training. '
                             'We set a large number by default as early stopping will be performed.')
    parser.add_argument('-p', '--result-path', type=str, required=True,
                        help='Path to training results')
    parser.add_argument('-t', '--tasks', default=None, type=str,
                        help='CSV column headers for the tasks to model. FOr multiple tasks, separate them by '
                             'comma, e.g., task1,task2,task3, ... If None, we will model '
                             'all the columns except for the smiles_column in the CSV file. '
                             '(default: None)')
    args = parser.parse_args().__dict__

    args['exp_name'] = '_'.join([args['model'], args['mode']])
    if args['tasks'] is not None:
        args['tasks'] = args['tasks'].split(',')
    args.update(configs[args['exp_name']])

    # Setup for experiments
    mkdir_p(args['result_path'])
