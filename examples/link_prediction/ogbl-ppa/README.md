# Link Prediction for ogbl-ppa

For a detailed description of the dataset, see [the OGB website](https://ogb.stanford.edu/docs/linkprop/).

## Models

- **Graph Convolutional Networks (GCN)** [1] 
- **GraphSAGE** [2]

## Dependencies

- **OGB v1.1.1**, which can be installed with ```pip install ogb```
- **DGL v0.4.3**

## Usage

To run with default options, simply do 

```bash
python full_graph_link_predictor.py
```

By default, we use CPU for computation as the graph is too large for a GPU with normal size.

The optional arguments are as follows:

```
--use_gpu, use gpu for computation
--use_sage, use GraphSAGE rather than GCN
--num_layers, number of GNN layers to use as well as linear layers for final link prediction (default=3)
--hidden_feats, size for hidden representations (default=256)
--dropout, (default=0.0)
--batch_size, batch size to use for link prediction (default=64 * 1024)
--lr, learning rate (default=0.01)
--epochs, number of epochs for training (default=20)
--eval_steps, evaluate hits@100 every {eval_steps} epochs (default=1)
--runs, number of random experiments to perform (default=1)
```

## Performance

For model evaluation, we consider hits@100 -- ranking each true link against 3,000,000 randomly-sampled 
negative edges, and counting the ratio of positive edges that are ranked at 100-th place or above.

Using the default parameters, the performance of 10 random runs is as follows.

| Method    | Train hits@100 | Validation hits@100 | Test hits@100 |
| --------- | -------------- | ------------------- | ------------- |
| GCN       | 12.87 ± 5.07   | 12.39 ± 4.85        | 11.65 ± 4.56  |
| GraphSAGE | 9.58 ± 0.99    | 9.44 ± 0.96         | 9.86 ± 1.21   |

| Method    | Average Time (hour) / epoch |
| --------- | --------------------------- |
| GCN       | 1.38                        |
| GraphSAGE | 1.47                        |

## References

[1] Kipf T N, Welling M. Semi-Supervised Classification with Graph Convolutional Networks[J]. 2016.

[2] Hamilton W L, Ying R, Leskovec J. Inductive Representation Learning on Large Graphs[J]. 2017.
