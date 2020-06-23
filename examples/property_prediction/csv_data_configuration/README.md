# Molecular Property Prediction on a New CSV Dataset

The scripts in this directory are helpful for quickly prototyping a GNN-based model for molecular property 
prediction on a new CSV dataset.

## Data Preparation

We assume that the molecular properties are recorded in a CSV file, where one column holds the SMILES strings 
of molecules and some other column(s) hold one or multiple molecular properties.

## Modeling

Currently we use a default setting as follows:
- Construct molecular graphs for each molecule, where nodes are atoms and edges are bonds
- Prepare initial atom features with [CanonicalAtomFeaturizer](https://lifesci.dgl.ai/generated/dgllife.utils.CanonicalAtomFeaturizer.html#dgllife.utils.CanonicalAtomFeaturizer).
- The dataset is split using scaffold split
- Perform early stopping and save the model that achieves the best validation performance
- Automatically perform hyperparameter search using Bayesian optimization

## Regression

To train a model for predicting real-valued properties, we can use 

```bash
python regression.py -c X -sc Y
```

where `X` specifies the path to the CSV file and `Y` specifies the header for the SMILES column in the CSV file.

Other optional arguments include:
- **Task**: `-t task1,task2,task3,...` can be used to specify the headers for task columns in the CSV file. If not specified, 
we assume all columns are molecular properties except for the SMILES column.
- **Split Ratio**: `-sr a,b,c` can be used to specify the proportion of the dataset to be used for training, validation and test. 
By default we use `0.8,0.1,0.1`.
- **Evaluation Metric**: `-me metric` can be used to specify the evaluation metric. 
By default we use `r2` for Pearson correlation coefficient. Alternatively, you can use `mae` for mean absolute error, 
and `rmse` for root mean square error.
- **Num Epochs**: `-n number` can be used to specify the maximum number of epochs for training. By default we set this to 
1000 as early stopping will be performed based on validation metric.
- **Print Every**: `-pe number` decides that the training progress will be printed every `number` minibatches. By default 
we set this to 20.
- **Result Path**: `-p path` specifies the path to save training results. By default, we set this to `regression_results`.
- **Number of Hyperparameter Search Trials**: `-ne num_trials` specifies the number of trials for hyperparameter search, 
default to 64.

Once the training is completed, the best hyperparameter configuration can be found in `regression_results/best_config.txt`, 
the model performance numbers and saved model can be found under `regression_results/best`.

## (Multi-label) Binary Classification

To train a model for predicting binary labels, we can use 

```bash
python classification.py -c X -sc Y
```

where `X` specifies the path to the CSV file and `Y` specifies the header for the SMILES column in the CSV file.

Other optional arguments include:
- **Task**: `-t task1,task2,task3,...` can be used to specify the headers for task columns in the CSV file. If not specified, 
we assume all columns are molecular properties except for the SMILES column.
- **Split Ratio**: `-sr a,b,c` can be used to specify the proportion of the dataset to be used for training, validation and test. 
By default we use `0.8,0.1,0.1`.
- **Evaluation Metric**: `-me metric` can be used to specify the evaluation metric. 
By default we use `roc_auc_score` for ROC-AUC score.
- **Num Epochs**: `-n number` can be used to specify the maximum number of epochs for training. By default we set this to 
1000 as early stopping will be performed based on validation metric.
- **Print Every**: `-pe number` decides that the training progress will be printed every `number` minibatches. By default 
we set this to 20.
- **Result Path**: `-p path` specifies the path to save training results. By default, we set this to `classification_results`.
- **Number of Hyperparameter Search Trials**: `-ne num_trials` specifies the number of trials for hyperparameter search, 
default to 64.

Once the training is completed, the best hyperparameter configuration can be found in `classification_results/best_config.txt`, 
the model performance numbers and saved model can be found under `classification_results/best`.
