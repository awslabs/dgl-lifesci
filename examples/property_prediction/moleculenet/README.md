# MoleculeNet

## Benchmark Introduction

[1] introduced the MoleculeNet benchmark, consisting of 17 datasets, which can be subdivided into four 
categories -- quantum mechanics, physical chemistry, biophysics and physiology.

## Tox21

### Dataset Introduction

The ["Toxicology in the 21st Century" (Tox21)](https://tripod.nih.gov/tox21/challenge/) initiative created
a public database measuring toxicity of compounds, which has been used in the 2014 Tox21 Data Challenge. The dataset
contains qualitative toxicity measurements for 8014 compounds on 12 different targets, including nuclear receptors and
stress response pathways. Each target yields a binary prediction problem. MoleculeNet [1] randomly splits the dataset
into training, validation and test set with a 80/10/10 ratio. By default we follow their split method.

### Training and Evaluation

We support three models for this dataset -- GCN [2], GAT [3] and Weave [4]. To train from scratch, run 

```bash
python classification.py -m X
```

where `X` can be `GCN`, `GAT`, or `Weave`.

You can also directly use the pre-trained models with

```bash
python classification.py -m X -p
```

The performance of pre-trained models is as follows:

| Model | Averaged Test ROC-AUC | 
| ----- | --------------------- |
| GCN   | 0.8326                |
| GAT   | 0.8528                |
| Weave | 0.8074                |

## References

[1] Wu et al. (2017) MoleculeNet: a benchmark for molecular machine learning. *Chemical Science* 9, 513-530.

[2] Kipf et al. (2017) Semi-Supervised Classification with Graph Convolutional Networks.
*The International Conference on Learning Representations (ICLR)*.

[3] Veličković et al. (2018) Graph Attention Networks. 
*The International Conference on Learning Representations (ICLR)*. 

[4] Kearnes et al. (2016) Molecular graph convolutions: moving beyond fingerprints. 
*Journal of Computer-Aided Molecular Design*.
