from dgllife.utils import atom_type_one_hot, atom_degree_one_hot, \
    atom_hybridization_one_hot, atom_is_aromatic_one_hot, \
    atom_chiral_tag_one_hot, atom_formal_charge_one_hot, atom_mass, \
    atom_implicit_valence_one_hot, BaseAtomFeaturizer, \
    ConcatFeaturizer, CanonicalBondFeaturizer
from functools import partial
from rdkit import Chem

atom_featurizer = BaseAtomFeaturizer(
    featurizer_funcs={
        'hv': ConcatFeaturizer(
            [partial(atom_degree_one_hot, allowable_set=[1, 2, 3, 4, 6]),
             partial(atom_type_one_hot, allowable_set=[
                 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Se', 'Si']),
             atom_chiral_tag_one_hot,
             partial(atom_formal_charge_one_hot, allowable_set=[-1, 0, 1]),
             partial(atom_hybridization_one_hot, allowable_set=[
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D2
             ]),
             partial(atom_implicit_valence_one_hot, allowable_set=list(range(4))),
             atom_is_aromatic_one_hot, atom_mass,
    ])}
)

if __name__ == '__main__':
    import pandas as pd

    from argparse import ArgumentParser
    from dgllife.data import MoleculeCSVDataset
    from dgllife.utils import smiles_to_bigraph, RandomSplitter

    from configure import configs
    from run import main
    from utils import mkdir_p, setup

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
    parser.add_argument('-s', '--smiles-column', type=str, default='smiles',
                        help='CSV column header for the SMIELS strings. (default: smiles)')
    parser.add_argument('-t', '--tasks', default=None, type=str,
                        help='CSV column headers for the tasks to model. For multiple tasks, separate them by '
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

    node_featurizer = atom_featurizer
    edge_featurizer = CanonicalBondFeaturizer(bond_data_field='he')
    df = pd.read_csv(args['csv_path'])
    dataset = MoleculeCSVDataset(
        df, smiles_to_bigraph,
        node_featurizer=node_featurizer,
        edge_featurizer=edge_featurizer,
        smiles_column=args['smiles_column'],
        cache_file_path=args['result_path'] + '/graph.bin',
        task_names=args['tasks']
    )
    args['tasks'] = dataset.task_names
    args = setup(args)
    train_set, val_set, test_set = RandomSplitter.train_val_test_split(
        dataset, frac_train=0.8, frac_val=0.1,
        frac_test=0.1, random_state=0)

    main(args, node_featurizer, edge_featurizer, train_set, val_set, test_set)
