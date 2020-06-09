# graph construction
from dgllife.utils import smiles_to_bigraph
# general featurization
from dgllife.utils import ConcatFeaturizer
# node featurization
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, \
    atom_formal_charge, atom_num_radical_electrons, \
    atom_hybridization_one_hot, atom_total_num_H_one_hot
# edge featurization
from dgllife.utils.featurizers import BaseBondFeaturizer
from functools import partial

from utils import chirality

attentivefp = {
    'random_seed': 8,
    'graph_feat_size': 200,
    'num_layers': 2,
    'num_timesteps': 2,
    'node_feat_size': 39,
    'edge_feat_size': 10,
    'n_tasks': 1,
    'dropout': 0.2,
    'weight_decay': 10 ** (-5.0),
    'lr': 10 ** (-2.5),
    'batch_size': 128,
    'num_epochs': 800,
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'patience': 80,
    'metric_name': 'rmse',
    'mode': 'lower',
    'smiles_to_graph': smiles_to_bigraph,
    # Follow the atom featurization in the original work
    'node_featurizer': BaseAtomFeaturizer(
        featurizer_funcs={'hv': ConcatFeaturizer([
            partial(atom_type_one_hot, allowable_set=[
                'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
                    encode_unknown=True),
            partial(atom_degree_one_hot, allowable_set=list(range(6))),
            atom_formal_charge, atom_num_radical_electrons,
            partial(atom_hybridization_one_hot, encode_unknown=True),
            lambda atom: [0], # A placeholder for aromatic information,
            atom_total_num_H_one_hot, chirality
        ],
        )}
    ),
    'edge_featurizer': BaseBondFeaturizer({
        'he': lambda bond: [0 for _ in range(10)]
    })
}

experiment_configures = {
    'AttentiveFP_Aromaticity': attentivefp
}
def get_exp_configure(exp_name):
    return experiment_configures[exp_name]
