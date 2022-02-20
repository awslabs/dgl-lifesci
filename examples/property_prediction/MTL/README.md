# Multitask Graph Neural Network for Molecular Property Prediction

[Paper](https://arxiv.org/abs/2111.13964)

## Usage

```
python -c CSV -m MODEL --mode MODE -p PATH -s SMILES -t TASKS
```

where:
- `CSV` specifies the path to a CSV file for the dataset
- `MODEL` specifies the model to use, which can be `GCN`, `GAT`, `MPNN`, or `AttentiveFP`
- `MODE` specifies the multitask architecture to use, which can be `parallel` or `bypass`
- `PATH` specifies the path to save training results
- `SMILES` specifies the SMIELS column header in the CSV file
- `TASKS` specifies the CSV column headers for the tasks to model. For multiple tasks, separate them by comma, e.g., task1,task2,task3. It not specified, all columns except for the SMILES column will be treated as properties/tasks.

## Example

For demonstration, you can generate a synthetic dataset as follows.

```python
import torch
import pandas as pd

data = {
    'smiles': ['CCO' for _ in range(128)],
    'logP': torch.randn(128).numpy().tolist(),
    'logD': torch.randn(128).numpy().tolist()
}
df = pd.DataFrame(data)
df.to_csv('syn_data.csv', index=False)
```

After you run an experiment with

```
python main.py -c syn_data.csv -m GCN --mode parallel -p results -s smiles -t logP,logD
```

Once the experiment is completed, `results/model.pth` is the trained model checkpoint
and `results/results.txt` is the evaluation result.

## Hyperparameters

The hyperparameters for all experiments are included in `configure.py`:

| Configure    | Multi-task  | Model | Architecture | Dataset |
| ------------ | ----------- | ----- | ------------ | ------- |
| GAT_parallel | Yes         | GAT   | Parallel     | All     |
| GAT_bypass   | Yes         | GAT   | Bypass       | All     |
| GAT_LogD     | No          | GAT   | Parallel     | LogD    |
| GAT_HLM      | No          | GAT   | Parallel     | HLM     |
| GAT_HH       | No          | GAT   | Parallel     | HH      |
| GAT_KinSol   | No          | GAT   | Parallel     | KinSol  |
| GCN_LogD     | No          | GCN   | Parallel     | LogD    |
| GCN_HLM      | No          | GCN   | Parallel     | HLM     |
| GCN_HH       | No          | GCN   | Parallel     | HH      |
| GCN_KinSol   | No          | GCN   | Parallel     | KinSol  |
| MPNN_LogD    | No          | MPNN  | Parallel     | LogD    |
| MPNN_HLM     | No          | MPNN  | Parallel     | HLM     |
| MPNN_HH      | No          | MPNN  | Parallel     | HH      |
| MPNN_KinSol  | No          | MPNN  | Parallel     | KinSol  |
| AttentiveFP_LogD    | No          | AttentiveFP  | Parallel     | LogD    |
| AttentiveFP_HLM     | No          | AttentiveFP  | Parallel     | HLM     |
| AttentiveFP_HH      | No          | AttentiveFP  | Parallel     | HH      |
| AttentiveFP_KinSol  | No          | AttentiveFP  | Parallel     | KinSol  |

## Cite

```
@article{https://doi.org/10.1002/minf.202100321,
author = {Broccatelli, Fabio and Trager, Richard and Reutlinger, Michael and Karypis, George and Li, Mufei},
title = {Benchmarking Accuracy and Generalizability of Four Graph Neural Networks Using Large In Vitro ADME Datasets from Different Chemical Spaces},
journal = {Molecular Informatics},
doi = {https://doi.org/10.1002/minf.202100321},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/minf.202100321},
}
```
