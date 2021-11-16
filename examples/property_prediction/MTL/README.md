# Multitask Graph Neural Network for Molecular Property Prediction

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
import pandas as pd

data = {
    'smiles': ['CCO' for _ in range(128)],
    'logP': [0.5 for _ in range(128)],
    'logD': [0.3 for _ in range(128)]
}
df = pd.DataFrame(data)
df.to_csv('syn_data.csv', index=False)
```

After you run an experiment with

```
python main.py -c syn_data.csv -m GCN --mode parallel -p results -s smiles -t logP,logD
```
