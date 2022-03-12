Installation
============

System requirements
-------------------
DGL-LifeSci should work on:

* Ubuntu 16.04
* macOS X
* Windows 10

DGL-LifeSci requires:

* Python 3.6 or later
* `DGL 0.4.3 or later <https://www.dgl.ai/pages/start.html>`_
* `PyTorch 1.2.0 or later <https://pytorch.org/>`_

Additionally, we require **RDKit 2018.09.3** for cheminformatics. We recommend installing it with

.. code:: bash

    conda install -c conda-forge rdkit==2018.09.3

Other verions of RDKit are not tested.

Install from pip
----------------

.. code:: bash

    pip install dgllife

.. _install-from-source:

Install from source
-------------------

To use the latest experimental features,

.. code:: bash

    git clone https://github.com/awslabs/dgl-lifesci.git
    cd dgl-lifesci/python
    python setup.py install
