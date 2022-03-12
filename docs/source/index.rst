DGL-LifeSci: Bringing Graph Neural Networks to Chemistry and Biology
===========================================================================================

DGL-LifeSci is a python package for applying graph neural networks to various tasks in chemistry
and biology, on top of PyTorch, DGL, and RDKit. It covers various applications, including:

* Molecular property prediction
* Generative models
* Reaction prediction
* Protein-ligand binding affinity prediction

.. toctree::
   :maxdepth: 1
   :caption: Installation
   :hidden:
   :glob:

   install/index
   cli

.. toctree::
    :maxdepth: 2
    :caption: API Reference
    :hidden:
    :glob:

    api/utils.mols
    api/utils.splitters
    api/utils.pipeline
    api/utils.complexes
    api/data
    api/model.pretrain
    api/model.gnn
    api/model.readout
    api/model.zoo

Free software
-------------
DGL-LifeSci is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions. Join us on `GitHub <https://github.com/awslabs/dgl-lifesci>`_.
