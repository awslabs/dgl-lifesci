# ogbg-ppa

For a detailed description of the dataset, see [the OGB website](https://ogb.stanford.edu/docs/graphprop/).

## Models

- **Graph Convolutional Networks (GCN)** [1] 
- **Graph Isomorphism Networks (GIN)** [2]

## Dependencies

- **OGB v1.2.1**, which can be installed with ```pip install ogb```

## Usage

To run the script, 

```bash
python main.py --gnn X
```

where `X` can be `gcn`, `gin`, `gcn-virtual` and `gin-virtual`. The postfix `-virtual` means that 
we will use a virtual node connected to all nodes in the graph for synchronizing information across all nodes.

By default, we use GPU whenever possible.

The optional arguments are as follows:

```
--dropout, dropout to use, (default=0.5)
--n_layers, number of GNN layers to use, (default=5)
--hidden_feats, number of hidden units in GNNs, (default=300)
--batch_size, batch size for training, (default=32)
--epochs, number of epochs for training, (default=100)
--num_workers, number of processes for data loading, (default=1)
--filename, filename to output results. By default, it will be the same as the gnn used.
```

## Performance

Using the default parameters, the performance of 10 random runs is as follows.

| Method      | Accuracy (%)  |
| ----------- | ------------- |
| GCN         | 67.80 +- 0.49 |
| GIN         | 69.31 +- 1.94 |
| GCN-virtual | 69.02 +- 0.47 |
| GIN-virtual | 70.62 +- 0.70 |

## References

[1] Kipf T., Welling M. Semi-Supervised Classification with Graph Convolutional Networks. 2016.

[2] Xu K., Hu W., Leskovec J., Jegelka S. How Powerful are Graph Neural Networks? 2019.
