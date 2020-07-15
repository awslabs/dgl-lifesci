# Hyperparameter Configuration

To manually set the hyperparameters for a model, modify the corresponding json file.

## Common Hyperparameters

- `lr`: (float) Learning rate for updating model parameters
- `weight_decay`: (float) Strength for L2 penalty in the objective function
- `patience`: (int) Number of epochs to wait before early stopping when validation performance no longer gets improved
- `batch_size`: (int) Batch size for mini-batch training

## GCN

- `gnn_hidden_feats`: (int) Hidden size for GNN layers
- `predictor_hidden_feats`: (int) Hidden size for the MLP predictor
- `num_gnn_layers`: (int) Number of GCN layers to use
- `residual`: (bool) Whether to use residual connection for each GCN layer
- `batchnorm`: (bool) Whether to apply batch normalization to the output of each GCN layer
- `dropout`: (float) Dropout probability

## GAT

- `gnn_hidden_feats`: (int) Hidden size for each attention head in GNN layers
- `num_heads`: (int) Number of attention heads in each GNN layer
- `alpha`: (float) Slope for negative values in LeakyReLU
- `predictor_hidden_feats`: (int) Hidden size for the MLP predictor
- `num_gnn_layers`: (int) Number of GNN layers to use
- `residual`: (bool) Whether to use residual connection for each GAT layer
- `dropout`: (float) Dropout probability

## Weave

- `num_gnn_layers`: (int) Number of GNN layers to use
- `gnn_hidden_feats`: (int) Hidden size for GNN layers
- `graph_feats`: (int) Hidden size for the MLP predictor
- `gaussian_expand`: (bool) Whether to expand each dimension of node features by 
gaussian histogram in computing graph representations.

## MPNN

- `node_out_feats`: (int) Hidden size for node representations in GNN layers
- `edge_hidden_feats`: (int) Hidden size for edge representations in GNN layers
- `num_step_message_passing` (int) Number of times for message passing, which is equivalent to the number of GNN layers
- `num_step_set2set`: (int) Number of set2set steps
- `num_layer_set2set`: (int) Number of set2set layers

## AttentiveFP

- `num_layers`: (int) Number of GNN layers
- `num_timesteps`: (int) Times of updating graph representations with GRU
- `graph_feat_size` (int) Hidden size for the graph representations
- `dropout`: (float) Dropout probability

## gin_supervised_contextpred / gin_supervised_edgepred / gin_supervised_infomax / gin_supervised_masking

- `jk`: (str) The way to aggregate the output of all GNN layers. One of `'concat'`, `'last'`, `'max'`, 
`'sum'`, separately for taking the concatenation of all GNN layer output, taking the output of the last 
GNN layer, performing max pooling across all GNN layer output, and summing all GNN layer output.
- `readout`: (str) The way to compute graph-level representations out of node-level representations, which 
can be one of `'sum'`, `'mean'`, `'max'`, and `'attention'`.
