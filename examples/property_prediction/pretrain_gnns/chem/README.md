# Strategies for Pre-training Graph Neural Networks

## Intro
This is a DGL implementation of the following paper based on PyTorch.

- [Strategies for Pre-training Graph Neural Networks.](https://arxiv.org/abs/1905.12265) W. Hu*, B. Liu*, J. Gomes, M. Zitnik., P. Liang, V. Pande, J. Leskovec International Conference on Learning Representations (ICLR), 2020.

## Dataset

## Usage
**1. Self-supervised pre-training**
``` python
python pretrain_masking.py  --output_model_file OUTPUT_MODEL_PATH
python pretrain_contextpred.py  --output_model_file OUTPUT_MODEL_PATH # to-do
```
**2. Supervised pre-training**
``` python
python pretrain_supervised.py --output_model_file OUTPUT_MODEL_PATH --input_model_file INPUT_MODEL_PATH
```
## Experiment Results
