# DGL-LifeSci

[Documentation](https://lifesci.dgl.ai/index.html) | [Discussion Forum](https://discuss.dgl.ai)

We also have a **slack channel** for real-time discussion. If you want to join the channel, contact mufeili1996@gmail.com.

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

DGL-LifeSci requires python 3.6+, DGL 0.5.2+ and PyTorch 1.5.0+.

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
# 0.2.8
```

### Cite

If you use DGL-LifeSci in a scientific publication, we would appreciate citations to the following paper:

```bash
@article{dgllife,
    title={DGL-LifeSci: An Open-Source Toolkit for Deep Learning on Graphs in Life Science},
    author={Mufei Li, Jinjing Zhou, Jiajing Hu, Wenxuan Fan, Yangkang Zhang, Yaxin Gu, George Karypis},
    year={2021},
    journal={arXiv preprint arXiv:2106.14232}
}
```
