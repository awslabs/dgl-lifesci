# ogbg-mol

For a detailed description of the dataset, see [the OGB website](https://ogb.stanford.edu/docs/graphprop/).
We just focus on hiv dataset and pcba dataset.

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

Hiv_dataset
| Method      |      ROC-AUC     |
| ----------- |  --------------- |
| GCN         | 0.7560 +- 0.0178 |
| GIN         | 0.7599 +- 0.0178 |
| GCN-virtual | 0.7599 +- 0.0178 |
| GIN-virtual | 0.7598 +- 0.0178 |

PCBA_dataset
| Method      |      PRC-AUC     |
| ----------- |  --------------- |
| GCN         | 0.1977 +- 0.0026 |
| GIN         | 0.2214 +- 0.0046 |
| GCN-virtual | 0.2395 +- 0.0042 |
| GIN-virtual | 0.2672 +- 0.0028 |

## References

[1] Kipf T., Welling M. Semi-Supervised Classification with Graph Convolutional Networks. 2016.

[2] Xu K., Hu W., Leskovec J., Jegelka S. How Powerful are Graph Neural Networks? 2019.
