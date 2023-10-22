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
--use_node_embedding, prepare node embeddings using node2vec
--num_layers, number of GNN layers to use as well as linear layers for final link prediction (default=3)
--hidden_feats, size for hidden representations (default=256)
--dropout, (default=0.0)
--batch_size, batch size to use for link prediction (default=64 * 1024)
--lr, learning rate (default=0.01)
--epochs, number of epochs for training (default=20)
--eval_steps, evaluate hits@100 every {eval_steps} epochs (default=1)
--runs, number of random experiments to perform (default=1)
```


Full-batch GCN training based on Node2Vec features.
To generate Node2Vec features, please run ```python node2vec.py```. This script requires node embeddings be saved in ```embedding.pt```. 

The optional arguments are as follows:

```
--embedding_dim, the size of each embedding vector (default=128)
--walk_length, the walk length (default=40)
--context_size, the actual context size which is considered for positive samples (default=20)
--walks_per_node, the number of walks to sample for each node (default=10)
--batch_size, batch size to use for sampling (default=256)
--lr, learning rate (default=0.01)
--epochs, number of epochs for training (default=2)
--log_steps, number of steps log (default=1)
```


## Performance

For model evaluation, we consider hits@100 -- ranking each true link against 3,000,000 randomly-sampled 
negative edges, and counting the ratio of positive edges that are ranked at 100-th place or above.

Using the default parameters, the performance of 10 random runs is as follows.

| Method       | Train hits@100 | Validation hits@100 | Test hits@100 |
| -----------  | -------------- | ------------------- | ------------- |
| GCN          | 23.95 ± 2.80   | 22.60 ± 2.59        | 21.30 ± 3.41  |
| GraphSAGE    | 13.88 ± 1.73   | 13.06 ± 1.51        | 11.90 ± 1.34  |
| Node2vec+GCN | 27.98 ± 2.63   | 26.45 ± 2.49        | 25.81 ± 2.58  |

| Method       | Average Time (hour) / epoch |
| -----------  | --------------------------- |
| GCN          | 1.25                        |
| GraphSAGE    | 1.28                        |
| Node2vec+GCN | 1.29                        |

## References

[1] Kipf T., Welling M. Semi-Supervised Classification with Graph Convolutional Networks. 2016.

[2] Hamilton W., Ying R., Leskovec J. Inductive Representation Learning on Large Graphs. 2017.
