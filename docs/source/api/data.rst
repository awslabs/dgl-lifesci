.. _apidata:

Datasets
========

.. contents:: Contents
    :local:

Molecular Property Prediction
-----------------------------

Tox21
`````

.. autoclass:: dgllife.data.Tox21
    :members: task_pos_weights, __getitem__, __len__
    :show-inheritance:

ESOL
````

.. autoclass:: dgllife.data.ESOL
    :members: __getitem__, __len__
    :show-inheritance:

FreeSolv
````````

.. autoclass:: dgllife.data.FreeSolv
    :members: __getitem__, __len__
    :show-inheritance:

Lipophilicity
`````````````

.. autoclass:: dgllife.data.Lipophilicity
    :members: __getitem__, __len__
    :show-inheritance:

Experimental solubility determined at AstraZeneca, extracted from ChEMBL
````````````````````````````````````````````````````````````````````````

.. autoclass:: dgllife.data.AstraZenecaChEMBLSolubility
    :members: __getitem__, __len__
    :show-inheritance:

Alchemy for Quantum Chemistry
`````````````````````````````

.. autoclass:: dgllife.data.TencentAlchemyDataset
    :members: set_mean_and_std, __getitem__, __len__

Pubmed Aromaticity
``````````````````

.. autoclass:: dgllife.data.PubChemBioAssayAromaticity
    :members: __getitem__, __len__
    :show-inheritance:

Adapting to New Datasets with CSV
`````````````````````````````````

.. autoclass:: dgllife.data.MoleculeCSVDataset
    :members: __getitem__, __len__

Adapting to New Datasets for Inference
``````````````````````````````````````

.. autoclass:: dgllife.data.UnlabeledSMILES
    :members: __getitem__, __len__

Reaction Prediction
-------------------

USPTO
`````

.. autoclass:: dgllife.data.USPTOCenter
    :members: __getitem__, __len__
    :show-inheritance:

.. autoclass:: dgllife.data.USPTORank
    :members: ignore_large, __getitem__, __len__
    :show-inheritance:

Adapting to New Datasets for Weisfeiler-Lehman Networks
```````````````````````````````````````````````````````

.. autoclass:: dgllife.data.WLNCenterDataset
    :members: __getitem__, __len__

.. autoclass:: dgllife.data.WLNRankDataset
    :members: ignore_large, __getitem__, __len__

Protein-Ligand Binding Affinity Prediction
------------------------------------------

PDBBind
```````

.. autoclass:: dgllife.data.PDBBind
    :members: __getitem__, __len__
