from hyperopt import hp

common_hyperparameters = {
    'lr': hp.uniform('lr', low=3e-4, high=1e-2),
    'weight_decay': hp.uniform('weight_decay', low=0, high=1e-4),
    'patience': hp.choice('patience', [30, 50, 100]),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512]),
    'dropout': hp.uniform('dropout', low=0., high=0.15)
}

gcn_hyperparameters = {
    'gnn_hidden_feats': hp.choice('gnn_hidden_feats', [32, 64, 128, 256]),
    'classifier_hidden_feats': hp.choice('classifier_hidden_feats', [16, 32, 64, 128]),
    'num_gnn_layers': hp.choice('num_gnn_layers', [1, 2, 3, 4, 5]),
    'residual': hp.choice('residual', [True, False]),
    'batch_norm': hp.choice('batchnorm', [True, False])
}

def init_hyper_space(model):
    """Initialize the hyperparameter search space

    Parameters
    ----------
    model : str
        Model for searching hyperparameters

    Returns
    -------
    dict
        Mapping hyperparameter names to the associated search spaces
    """
    candidate_hypers = dict()
    candidate_hypers.update(common_hyperparameters)
    if model == 'GCN':
        candidate_hypers.update(gcn_hyperparameters)
    else:
        return ValueError('Unexpected model: {}'.format(model))
    return candidate_hypers
