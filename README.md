# DGL-LifeSci

[Documentation](https://lifesci.dgl.ai/index.html) | [Discussion Forum](https://discuss.dgl.ai)

We also have a **slack channel** for real-time discussion. If you want to join the channel, contact mufeili1996@gmail.com.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  * [Requirements](#requirements)
  * [Pip installation for DGL-LifeSci](#pip-installation-for-dgl-lifesci)
  * [Installation from source](#installation-from-source)
  * [Verifying successful installation](#verifying-successful-installation)
- [Command Line Interface](#command-line-interface)
- [Cite](#cite)

## Introduction

Deep learning on graphs has been an arising trend in the past few years. There are a lot of graphs in
life science such as molecular graphs and biological networks, making it an import area for applying
deep learning on graphs. DGL-LifeSci is a DGL-based package for various applications in life science
with graph neural networks.

We provide various functionalities, including but not limited to methods for graph construction,
featurization, and evaluation, model architectures, training scripts and pre-trained models.

For a list of community contributors, see [here](CONTRIBUTORS.md).

**For a full list of work implemented in DGL-LifeSci, see [here](examples/README.md).**

## Installation

### Requirements

DGL-LifeSci should work on

* all Linux distributions no earlier than Ubuntu 16.04
* macOS X
* Windows 10

It is recommended to create a conda environment for DGL-LifeSci with for example

```
conda create -n dgllife python=3.6
```

DGL-LifeSci requires python 3.6+, DGL 0.7.0+ and PyTorch 1.5.0+.

[Install pytorch](https://pytorch.org/get-started/locally/)

[Install dgl](https://www.dgl.ai/pages/start.html)


Additionally, we require `RDKit 2018.09.3` for utils related to cheminformatics. We recommend installing it with

```
conda install -c rdkit rdkit==2018.09.3
```

For other installation recipes for RDKit, see the [official documentation](https://www.rdkit.org/docs/Install.html).

### Pip installation for DGL-LifeSci

```
pip install dgllife
```

### Installation from source

If you want to try experimental features, you can install from source as follows:

```
git clone https://github.com/awslabs/dgl-lifesci.git
cd dgl-lifesci/python
python setup.py install
```

### Verifying successful installation

Once you have installed the package, you can verify the success of installation with

```python
import dgllife

print(dgllife.__version__)
# 0.2.9
```

## Command Line Interface

DGL-LifeSci provides command line interfaces that allow users to perform modeling without any background in programming and deep learning:

- [Molecular Property Prediction](examples/property_prediction/csv_data_configuration/)
- [Reaction Prediction](examples/reaction_prediction/rexgen_direct/)

## Cite

If you use DGL-LifeSci in a scientific publication, we would appreciate citations to the following paper:

```
@article{dgllife,
    title={DGL-LifeSci: An Open-Source Toolkit for Deep Learning on Graphs in Life Science},
    author={Mufei Li and Jinjing Zhou and Jiajing Hu and Wenxuan Fan and Yangkang Zhang and Yaxin Gu and George Karypis},
    year={2021},
    journal = {ACS Omega}
}
```
