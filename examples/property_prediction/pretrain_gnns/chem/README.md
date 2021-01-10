# Strategies for Pre-training Graph Neural Networks

## Intro
This is a DGL implementation of the following paper based on PyTorch.

- [Strategies for Pre-training Graph Neural Networks.](https://arxiv.org/abs/1905.12265) W. Hu*, B. Liu*, J. Gomes, M. Zitnik., P. Liang, V. Pande, J. Leskovec. *International Conference on Learning Representations (ICLR)*, 2020.

## Datasets
- For node-level self-supervised pre-training, 2 million unlabeled molecules sampled from the ZINC15 database are used. Custom datasets are supported.
- For graph-level multi-task supervised pre-training, a preprocessed ChEMBL dataset is used, which contains 456K molecules with 1310 kinds of diverse and extensive biochemical assays. Custom datasets are supported.
- For fine-tuning downstream tasks, BBBP, Tox21, ToxCast, SIDER, MUV, HIV and BACE dataset are supported.

## Usage
**1. Self-supervised pre-training**

This paper purposed an attribute masking pre-training method. It randomly masks input node/edge attributes by replacing them with special masked indicators, then the GNN will predict those attributes based on neighboring structure.

``` bash
python pretrain_masking.py  --output_model_file OUTPUT_MODEL_FILE
```
The self-supervised pre-training model will be found in `OUTPUT_MODEL_FILE` after training (default filename: pretrain_masking.pth).

If a custom dataset is specified, the path needs to be provided with `--dataset`. The custom dataset is supposed to be a text file, where every line is a molecule SMILES except that the first is 'smiles'.

**2. Supervised pre-training**
``` bash
python pretrain_supervised.py --input_model_file INPUT_MODEL_FILE --output_model_file OUTPUT_MODEL_FILE
```
The self-supervised pre-trained model can be loaded from `INPUT_MODEL_FILE`.

The supervised pre-training model will be found in `OUTPUT_MODEL_FILE` after training (default filename: pretrain_supervised.pth).

If a custom dataset is specified, the path needs to be provided with `--dataset`. The custom dataset is supposed to be a `.pkl` file, which is pickled from "a list of tuples". The first element in every `tuple` should be a molecule SMILES in class `str`, and the second element should be its corresponding label in class `torch.Tensor`. Possible values are {-1, 0, 1} in labels. "1" means positive, and "-1" means negative. "0" indicates the molecule is invalid.

**3. Fine-tuning for downstream dataset**
``` bash
python classification.py --input_model_file INPUT_MODEL_FILE --output_model_file OUTPUT_MODEL_FILE --dataset DOWNSTREAM_DATASET
```

The supervised pre-trained model can be loaded from `INPUT_MODEL_FILE`.

The fine-tuned model will be found in `OUTPUT_MODEL_FILE` after training (default filename: pretrain_fine_tuning.pth).

## Experiment Results

With the default parameters, following downstream task results are based on Attribute Masking (Node-level) and Supervised (Graph-level) pre-training strategy with GIN.

| Dataset  | ROC-AUC (%) |  ROC-AUC reported (%)   |
| :-----: | :-----: | :--------: |
|  BBBP   |  71.75  | 66.5 ± 2.5 |
|  Tox21  |  72.67  | 77.9 ± 0.4 |
| ToxCast |  62.22  | 65.1 ± 0.3 |
|  SIDER  |  58.97  | 63.9 ± 0.9 |
|   MUV   |  79.44  | 81.2 ± 1.9 |
|   HIV   |  74.52  | 77.1 ± 1.2 |
|  BACE   |  77.34  | 80.3 ± 0.9 |
