# Strategies for Pre-training Graph Neural Networks

## Intro
This is a DGL implementation of the following paper based on PyTorch.

- [Strategies for Pre-training Graph Neural Networks.](https://arxiv.org/abs/1905.12265) W. Hu*, B. Liu*, J. Gomes, M. Zitnik., P. Liang, V. Pande, J. Leskovec. International Conference on Learning Representations (ICLR), 2020.

## Dataset
- to-do

## Usage
**1. Self-supervised pre-training**
``` bash
python pretrain_masking.py  --output_model_file OUTPUT_MODEL_PATH
```
The self-supervised pre-training model will be found in `OUTPUT_MODEL_PATH` after training. (default filename: pretrain_masking.pth)

**2. Supervised pre-training**
``` bash
python pretrain_supervised.py --output_model_file OUTPUT_MODEL_PATH --input_model_file INPUT_MODEL_PATH
```
The supervised pre-training model will be found in `OUTPUT_MODEL_PATH` after training. (default filename: pretrain_supervised.pth)

## Experiment Results
- to-do