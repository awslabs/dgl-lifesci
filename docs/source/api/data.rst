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
    :members: __getitem__, __len__, task_pos_weights
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

PCBA
````

.. autoclass:: dgllife.data.PCBA
    :members: __getitem__, __len__, task_pos_weights
    :show-inheritance:

MUV
```

.. autoclass:: dgllife.data.MUV
    :members: __getitem__, __len__, task_pos_weights
    :show-inheritance:

HIV
```

.. autoclass:: dgllife.data.HIV
    :members: __getitem__, __len__, task_pos_weights
    :show-inheritance:

BACE
````

.. autoclass:: dgllife.data.BACE
    :members: __getitem__, __len__, task_pos_weights
    :show-inheritance:

BBBP
````

.. autoclass:: dgllife.data.BBBP
    :members: __getitem__, __len__, task_pos_weights
    :show-inheritance:

ToxCast
```````

.. autoclass:: dgllife.data.ToxCast
    :members: __getitem__, __len__, task_pos_weights
    :show-inheritance:

SIDER
`````

.. autoclass:: dgllife.data.SIDER
    :members: __getitem__, __len__, task_pos_weights
    :show-inheritance:

ClinTox
```````

.. autoclass:: dgllife.data.ClinTox
    :members: __getitem__, __len__, task_pos_weights
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
    :members: __getitem__, __len__, task_pos_weights

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

Generative Models
-----------------

JTVAE
`````

.. autoclass:: dgllife.data.DGLMolTree
    :members: treesize, recover, assemble

.. autoclass:: dgllife.data.JTVAEDataset
    :members: __len__, __getitem__, move_to_device

.. autoclass:: dgllife.data.JTVAECollator
    :members: __call__

Protein-Ligand Binding Affinity Prediction
------------------------------------------

PDBBind
```````

.. autoclass:: dgllife.data.PDBBind
    :members: __getitem__, __len__
