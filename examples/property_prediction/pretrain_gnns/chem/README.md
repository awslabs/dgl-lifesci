# Strategies for Pre-training Graph Neural Networks

## Intro
This is a DGL implementation of the following paper based on PyTorch.

- [Strategies for Pre-training Graph Neural Networks.](https://arxiv.org/abs/1905.12265) W. Hu*, B. Liu*, J. Gomes, M. Zitnik., P. Liang, V. Pande, J. Leskovec. *International Conference on Learning Representations (ICLR)*, 2020.

## Datasets
- For node-level self-supervised pre-training, 2 million unlabeled molecules sampled from the ZINC15 database are used. Custom datasets are supported.
- For graph-level multi-task supervised pre-training, a preprocessed ChEMBL dataset is used, which contains 456K molecules with 1310 kinds of diverse and extensive biochemical assays. Custom datasets are supported.

## Usage
**1. Self-supervised pre-training**

This paper purposed an attribute masking pre-training method. It randomly mask input node/edge attributes by replacing them with special masked indicators, then the GNN will predict those attributes based on neighboring structure.

``` bash
python pretrain_masking.py  --output_model_file OUTPUT_MODEL_PATH
```
The self-supervised pre-training model will be found in `OUTPUT_MODEL_PATH` after training (default filename: pretrain_masking.pth).

If a custom dataset is specified, the path need to be provided with `--dataset`. The custom dataset is supposed to be a text file, which every line is a molecule SMILES. 

**2. Supervised pre-training**
``` bash
python pretrain_supervised.py --output_model_file OUTPUT_MODEL_PATH --input_model_file INPUT_MODEL_PATH
```
The supervised pre-training model will be found in `OUTPUT_MODEL_PATH` after training (default filename: pretrain_supervised.pth).

If a custom dataset is specified, the path needs to be provided with `--dataset`. The custom dataset is supposed to be a `.pkl` file, which is pickled from "a list of tuples". The first element in every `tuple` should be a molecule SMILES in class `str`, and the second element should be its corresponding label in class `torch.Tensor`. Possible values are {-1, 0, 1} in labels. "1" means positive, and "-1" means negative. "0" indicates the molecule is invalid.

## Experiment Results
- to-do