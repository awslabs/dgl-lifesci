# MoleculeNet

[MoleculeNet](https://arxiv.org/abs/1703.00564) is a popular benchmark for molecular machine learning.
For validating the robustness of DGL-LifeSci, we report the results of applying the 
[command-line interface](../csv_data_configuration) against 12 datasets in MoleculeNet, including 
9 classification datasets (BACE, BBBP, ClinTox, HIV, MUV, PCBA, SIDER, Tox21 and ToxCast) and 
3 regression datasets (ESOL, FreeSolv and Lipophilicity). 

For all these datasets, we use a scaffold split and perform a hyperparameter search with 
Bayesian Optimization for 32 trials. For each trial, we train a randomly initialized model
until the validation performance no longer gets improved for a number of epochs. The model
achieving the best validation performance across trials is then evaluated on the test set.

To train a model on a classification dataset, use 

```bash
python classification.py -d DATASET -mo MODEL -f FEATURE 
```

where:
- `DATASET` specifies the dataset for experiment, which can be one of `MUV`, `BACE`, `BBBP`, 
  `ClinTox`, `SIDER`, `ToxCast`, `HIV`, `PCBA`, `Tox21`
- `MODEL` specifies the model to use, which can be one of `GCN`, `GAT`, `Weave`, `MPNN`, 
  `AttentiveFP`, `gin_supervised_contextpred`, `gin_supervised_infomax`, 
  `gin_supervised_edgepred`, `gin_supervised_masking`, `NF`
- `FEATURE ` specifies the featurization method to use, which can be one of `canonical`, 
  `attentivefp`. This argument can be omitted if the model is one of `gin_supervised_*`.
  
To train a model on a regression dataset, use

```bash
python regression.py -d DATASET -mo MODEL -f FEATURE 
```

where:
- `DATASET` specifies the dataset for experiment, which can be one of `FreeSolv`, 
  `Lipophilicity`, `ESOL`
- `MODEL` specifies the model to use, which can be one of `GCN`, `GAT`, `Weave`, `MPNN`, 
  `AttentiveFP`, `gin_supervised_contextpred`, `gin_supervised_infomax`, 
  `gin_supervised_edgepred`, `gin_supervised_masking`
- `FEATURE ` specifies the featurization method to use, which can be one of `canonical`, 
  `attentivefp`. This argument can be omitted if the model is one of `gin_supervised_*`.

For both cases above, you can directly evaluate a pre-trained model without training from
scratch by appending `-p`.

The performance of the pre-trained models can be found below.

## BACE

| method                     | Val ROC-AUC | Test ROC-AUC |
| -------------------------- | ----------- | ------------ |
| GCN + canonical            | 0.77        | 0.84         |
| GCN + attentivefp          | 0.75        | 0.81         |
| GAT + canonical            | 0.74        | 0.84         |
| GAT + attentivefp          | 0.76        | 0.82         |
| Weave + canonical          | 0.69        | 0.79         |
| Weave + attentivefp        | 0.70        | 0.76         |
| MPNN + canonical           | 0.77        | 0.85         |
| MPNN + attentivefp         | 0.73        | 0.71         |
| AttentiveFP + canonical    | 0.70        | 0.73         |
| AttentiveFP + attentivefp  | 0.72        | 0.79         |
| gin_supervised_contextpred | 0.73        | 0.86         |
| gin_supervised_infomax     | 0.74        | 0.71         |
| gin_supervised_edgepred    | 0.76        | 0.86         |
| gin_supervised_masking     | 0.71        | 0.74         |
| NF + canonical             | 0.71        | 0.80         |

## BBBP

| method                     | Val ROC-AUC | Test ROC-AUC |
| -------------------------- | ----------- | ------------ |
| GCN + canonical            | 0.98        | 0.63         |
| GCN + attentivefp          | 0.98        | 0.64         |
| GAT + canonical            | 0.98        | 0.68         |
| GAT + attentivefp          | 0.97        | 0.69         |
| Weave + canonical          | 0.97        | 0.67         |
| Weave + attentivefp        | 0.97        | 0.69         |
| MPNN + canonical           | 0.96        | 0.65         |
| MPNN + attentivefp         | 0.97        | 0.68         |
| AttentiveFP + canonical    | 0.98        | 0.71         |
| AttentiveFP + attentivefp  | 0.97        | 0.65         |
| gin_supervised_contextpred | 0.95        | 0.63         |
| gin_supervised_infomax     | 0.97        | 0.72         |
| gin_supervised_edgepred    | 0.98        | 0.70         |
| gin_supervised_masking     | 0.97        | 0.72         |
| NF + canonical             | 0.98        | 0.66         |

## ClinTox

| method                     | Val ROC-AUC | Test ROC-AUC |
| -------------------------- | ----------- | ------------ |
| GCN + canonical            | 0.99        | 0.90         |
| GCN + attentivefp          | 0.99        | 0.85         |
| GAT + canonical            | 0.96        | 0.88         |
| GAT + attentivefp          | 0.98        | 0.82         |
| Weave + canonical          | 0.96        | 0.90         |
| Weave + attentivefp        | 0.96        | 0.90         |
| MPNN + canonical           | 0.99        | 0.84         |
| MPNN + attentivefp         | 0.99        | 0.82         |
| AttentiveFP + canonical    | 0.99        | 0.89         |
| AttentiveFP + attentivefp  | 0.97        | 0.85         |

## ESOL

| method                     | Val RMSE | Test RMSE |
| -------------------------- | -------- | --------- |
| GCN + canonical            | 0.83     | 0.86      |
| GCN + attentivefp          | 0.93     | 0.89      |
| GAT + canonical            | 0.99     | 0.94      |
| GAT + attentivefp          | 0.89     | 0.88      |
| Weave + canonical          | 0.95     | 1.00      |
| Weave + attentivefp        | 1.24     | 1.20      |
| MPNN + canonical           | 0.87     | 1.12      |
| MPNN + attentivefp         | 0.95     | 1.02      |
| AttentiveFP + canonical    | 0.86     | 0.82      |
| AttentiveFP + attentivefp  | 0.91     | 0.93      |
| gin_supervised_contextpred | 0.99     | 0.98      |
| gin_supervised_infomax     | 1.09     | 0.97      |
| gin_supervised_edgepred    | 1.29     | 1.50      | 
| gin_supervised_masking     | 1.09     | 1.18      |

## FreeSolv

| method                     | Val RMSE | Test RMSE |
| -------------------------- | -------- | --------- |
| GCN + canonical            | 1.67     | 3.12      |
| GCN + attentivefp          | 1.35     | 2.31      |
| GAT + canonical            | 2.57     | 2.73      |
| GAT + attentivefp          | 2.32     | 2.51      |
| Weave + canonical          | 1.71     | 3.13      |
| Weave + attentivefp        | 1.51     | 3.74      |
| MPNN + canonical           | 1.91     | 2.43      |
| MPNN + attentivefp         | 1.85     | 2.44      |
| AttentiveFP + canonical    | 1.67     | 3.26      |
| AttentiveFP + attentivefp  | 1.25     | 3.01      |
| gin_supervised_contextpred | 1.77     | 3.37      |
| gin_supervised_infomax     | 2.31     | 3.36      |
| gin_supervised_edgepred    | 2.25     | 2.49      |
| gin_supervised_masking     | 2.18     | 3.34      |

## HIV

| method                     | Val ROC-AUC | Test ROC-AUC |
| -------------------------- | ----------- | ------------ |
| GCN + canonical            | 0.83        | 0.76         |
| GCN + attentivefp          | 0.84        | 0.74         |
| GAT + canonical            | 0.81        | 0.76         |
| GAT + attentivefp          | 0.74        | 0.52         |
| Weave + canonical          | 0.76        | 0.73         |
| Weave + attentivefp        | 0.78        | 0.73         |
| MPNN + canonical           | 0.77        | 0.74         |
| MPNN + attentivefp         | 0.81        | 0.73         |
| AttentiveFP + canonical    | 0.76        | 0.75         |
| AttentiveFP + attentivefp  | 0.73        | 0.72         |
| gin_supervised_contextpred | 0.82        | 0.77         |
| gin_supervised_infomax     | 0.88        | 0.76         |
| gin_supervised_edgepred    | 0.80        | 0.72         |
| gin_supervised_masking     | 0.80        | 0.75         |
| NF + canonical             | 0.80        | 0.74         |

## Lipophilicity

| method                     | Val RMSE | Test RMSE |
| -------------------------- | -------- | --------- |
| GCN + canonical            | 0.74     | 0.76      |
| GCN + attentivefp          | 0.51     | 0.68      |
| GAT + canonical            | 0.81     | 0.85      |
| GAT + attentivefp          | 0.57     | 0.73      |
| Weave + canonical          | 0.83     | 0.80      |
| Weave + attentivefp        | 0.49     | 0.67      |
| MPNN + canonical           | 0.72     | 0.73      |
| MPNN + attentivefp         | 0.56     | 0.68      |
| AttentiveFP + canonical    | 0.69     | 0.73      |
| AttentiveFP + attentivefp  | 0.71     | 0.73      |
| gin_supervised_contextpred | 0.73     | 0.81      |
| gin_supervised_infomax     | 0.76     | 0.78      |
| gin_supervised_edgepred    | 0.75     | 0.79      |
| gin_supervised_masking     | 0.59     | 0.71      |

## MUV

| method                     | Val ROC-AUC | Test ROC-AUC |
| -------------------------- | ----------- | ------------ |
| GCN + canonical            | 0.69        | 0.61         |
| GCN + attentivefp          | 0.71        | 0.63         |
| GAT + canonical            | 0.69        | 0.64         |
| GAT + attentivefp          | 0.71        | 0.60         |
| Weave + canonical          | 0.71        | 0.60         |
| Weave + attentivefp        | 0.61        | 0.55         |
| MPNN + canonical           | 0.73        | 0.60         |
| MPNN + attentivefp         | 0.74        | 0.62         |
| AttentiveFP + canonical    | 0.70        | 0.61         |
| AttentiveFP + attentivefp  | 0.75        | 0.67         |
| gin_supervised_contextpred | 0.72        | 0.61         |
| gin_supervised_infomax     | 0.72        | 0.66         |
| gin_supervised_edgepred    | 0.71        | 0.61         |
| gin_supervised_masking     | 0.71        | 0.62         |

## PCBA

| method                     | Val ROC-AUC | Test ROC-AUC |
| -------------------------- | ----------- | ------------ |
| GCN + canonical            | 0.74        | 0.74         |
| GCN + attentivefp          | 0.75        | 0.75         |
| GAT + canonical            | 0.66        | 0.66         |
| GAT + attentivefp          | 0.62        | 0.62         |
| Weave + canonical          | 0.64        | 0.65         | 
| Weave + attentivefp        | 0.66        | 0.66         |
| MPNN + canonical           | 0.82        | 0.82         | 
| MPNN + attentivefp         | 0.72        | 0.72         | 
| AttentiveFP + canonical    | 0.69        | 0.68         |
| AttentiveFP + attentivefp  | 0.71        | 0.71         |
| gin_supervised_contextpred | 0.74        | 0.74         |
| gin_supervised_infomax     | 0.72        | 0.72         |
| gin_supervised_edgepred    | 0.62        | 0.62         |
| gin_supervised_masking     | 0.84        | 0.84         |

## SIDER

| method                     | Val ROC-AUC | Test ROC-AUC |
| -------------------------- | ----------- | ------------ |
| GCN + canonical            | 0.61        | 0.58         |
| GCN + attentivefp          | 0.65        | 0.62         |
| GAT + canonical            | 0.59        | 0.52         |
| GAT + attentivefp          | 0.57        | 0.52         |
| Weave + canonical          | 0.61        | 0.58         |
| Weave + attentivefp        | 0.56        | 0.62         |
| MPNN + canonical           | 0.57        | 0.54         |
| MPNN + attentivefp         | 0.56        | 0.50         |
| AttentiveFP + canonical    | 0.57        | 0.53         |
| AttentiveFP + attentivefp  | 0.57        | 0.49         |
| gin_supervised_contextpred | 0.60        | 0.61         |
| gin_supervised_infomax     | 0.61        | 0.63         |
| gin_supervised_edgepred    | 0.62        | 0.66         |
| gin_supervised_masking     | 0.61        | 0.58         |
| NF + canonical             | 0.58        | 0.53         |

## Tox21

| method                     | Val ROC-AUC | Test ROC-AUC |
| -------------------------- | ----------- | ------------ |
| GCN + canonical            | 0.82        | 0.77         |
| GCN + attentivefp          | 0.82        | 0.79         |
| GAT + canonical            | 0.73        | 0.71         |
| GAT + attentivefp          | 0.75        | 0.72         |
| Weave + canonical          | 0.56        | 0.56         |
| Weave + attentivefp        | 0.75        | 0.73         |
| MPNN + canonical           | 0.72        | 0.70         |
| MPNN + attentivefp         | 0.73        | 0.68         |
| AttentiveFP + canonical    | 0.73        | 0.70         |
| AttentiveFP + attentivefp  | 0.79        | 0.74         |
| gin_supervised_contextpred | 0.77        | 0.75         |
| gin_supervised_infomax     | 0.82        | 0.78         |
| gin_supervised_edgepred    | 0.83        | 0.80         |
| gin_supervised_masking     | 0.77        | 0.75         |
| NF + canonical             | 0.80        | 0.75         |

## ToxCast

| method                     | Val ROC-AUC | Test ROC-AUC |
| -------------------------- | ----------- | ------------ |
| GCN + canonical            | 0.64        | 0.62         |
| GCN + attentivefp          | 0.68        | 0.64         |
| GAT + canonical            | 0.66        | 0.64         |
| GAT + attentivefp          | 0.63        | 0.59         |
| Weave + canonical          | 0.63        | 0.62         |
| Weave + attentivefp        | 0.62        | 0.61         |
| MPNN + canonical           | 0.63        | 0.59         |
| MPNN + attentivefp         | 0.63        | 0.59         |
| AttentiveFP + canonical    | 0.62        | 0.57         |
| AttentiveFP + attentivefp  | 0.62        | 0.59         |
| gin_supervised_contextpred | 0.67        | 0.64         |
| gin_supervised_infomax     | 0.62        | 0.59         |
| gin_supervised_edgepred    | 0.61        | 0.59         |
| gin_supervised_masking     | 0.63        | 0.58         |
| NF + canonical             | 0.63        | 0.60         |
