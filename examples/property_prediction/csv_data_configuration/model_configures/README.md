# Hyperparameter Configuration

To manually set the hyperparameters for a model, modify the corresponding json file.

## Common Hyperparameters

- `lr`: (float) Learning rate for updating model parameters
- `weight_decay`: (float) Strength for L2 penalty in the objective function
- `patience`: (int) Number of epochs to wait before early stopping when validation performance no longer gets improved
- `batch_size`: (int) Batch size for mini-batch training
- `dropout`: (float) Dropout probability

## GCN

- `gnn_hidden_feats`: (int) Hidden size for GNN layers
- `predictor_hidden_feats`: (int) Hidden size for the MLP predictor
- `num_gnn_layers`: (int) Number of GCN layers to use
- `residual`: (bool) Whether to use residual connection for each GCN layer
- `batchnorm`: (bool) Whether to apply batch normalization to the output of each GCN layer
