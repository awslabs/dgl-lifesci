# Link Prediction for ogbl-ppa


## Models

- **Graph Convolutional Networks (GCN)** [1]: Semi-Supervised Classification with Graph Convolutional Networks 
- **Graph SAmple and aggreGatE (GraphSAGE)** [2]: Inductive Representation Learning on Large Graphs

## Dependencies

- **OGB v1.1.1**
- **DGL v0.4.3**

## Usage

Use `full_graph_link_predictor.py` with arguments
```
--device, Device to use (default=0)
--log_steps, (default=1)
--use_node_embedding, Whether to use node embedding (action='store_true')
--use_sage, Whether to use GraphSAGE model (action='store_true')
--num_layers, (default=3)
--hidden_channels, (default=256)
--dropout, (default=0.0)
--batch_size, (default=64 * 1024)
--lr, Learning rate (default=0.01)
--epochs, (default=20)
--eval_steps, (default=1)
--runs, (default=1)
```

## Performance

Using the default parameters, the performance of the two models on the ogbl-ppa dataset(Hits=100):

| Method  | %Training@Hits | %Validation@Hits | %Test@Hits |
| ------- | ---------------- | -------- | ------- |
| GCN | 12.87±5.07  | 12.39±4.85| 11.65±4.56 |
| GraphSAGE| 9.58±0.99| 9.44±0.96| 9.86±1.21|


| Method  | Average Time/epoch |
| ------- | -------------------------- |
| GCN | 1:23:12.86 |
| GraphSAGE| 1:28:49:46|

