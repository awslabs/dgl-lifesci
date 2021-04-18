# Junction Tree Variational Autoencoder for Molecular Graph Generation

- [paper](https://arxiv.org/abs/1802.04364)
- [authors' code](https://github.com/wengong-jin/icml18-jtnn/tree/master/molvae)

## Training

We trained VAE model in two phases:

1. We first train a model without KL regularization term. The model checkpoints will be saved as `pre_model/model.iter-x`, where `x + 1` is the number of training epochs.
2. We then train the model with KL regularization by passing the path to a saved model checkpoint. The model checkpoints will be saved as `vae_model/model.iter-x`.

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py
CUDA_VISIBLE_DEVICES=0 python vaetrain.py -m pre_model/model.iter-2
```

Note that the weight of the KL regularization term generally controls a trade off between reconstruction accuracy and generation diversity. To adjust the weight, specify `-z` (default: 0.001).

## Testing

For molecule reconstruction, 

```bash
CUDA_VISIBLE_DEVICES=0 python reconstruct.py -m Y
```

where `Y` is the path to a model checkpoint. If not specified, this will evaluate on a pre-trained model trained without KL regularization.
