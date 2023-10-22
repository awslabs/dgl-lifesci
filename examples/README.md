# Work Implemented in DGL-LifeSci

We provide various examples across 3 applications -- property prediction, generative models and protein-ligand binding affinity prediction.

## Datasets/Benchmarks

- MoleculeNet: A Benchmark for Molecular Machine Learning [[paper]](https://arxiv.org/abs/1703.00564), [[website]](http://moleculenet.ai/)
    - [ESOL with DGL](../python/dgllife/data/esol.py)
    - [FreeSolv with DGL](../python/dgllife/data/freesolv.py)
    - [Lipophilicity with DGL](../python/dgllife/data/lipophilicity.py)
    - [Tox21 with DGL](../python/dgllife/data/tox21.py)
    - [PDBBind with DGL](../python/dgllife/data/pdbbind.py)
- Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models [[paper]](https://arxiv.org/abs/1906.09427), [[github]](https://github.com/tencent-alchemy/Alchemy)
    - [Alchemy with DGL](../python/dgllife/data/alchemy.py)
- PubChem Aromaticity [[paper]](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959)
    - [PubChem Aromaticity with DGL](../python/dgllife/data/pubchem_aromaticity.py)
- OGB [[paper]](https://arxiv.org/abs/2005.00687)
    - [ogbl-ppa](link_prediction/ogbl-ppa)
- AstraZeneca Experimental Solubility from ChEMBL [[record]](https://www.ebi.ac.uk/chembl/document_report_card/CHEMBL3301361/)
    - [Dataset](../python/dgllife/data/astrazeneca_chembl_solubility.py)

## Property Prediction

- Molecular graph convolutions: moving beyond fingerprints (Weave) [[paper]](https://arxiv.org/abs/1603.00856), [[github]](https://github.com/deepchem/deepchem)
    - [Weave Predictor with DGL](../python/dgllife/model/model_zoo/weave_predictor.py)
    - [Example for Molecule Classification](property_prediction/moleculenet/classification.py)
- Semi-Supervised Classification with Graph Convolutional Networks (GCN) [[paper]](https://arxiv.org/abs/1609.02907), [[github]](https://github.com/tkipf/gcn)
    - [GCN-Based Predictor with DGL](../python/dgllife/model/model_zoo/gcn_predictor.py)
    - [Example for Molecule Classification](property_prediction/moleculenet/classification.py)
- Graph Attention Networks (GAT) [[paper]](https://arxiv.org/abs/1710.10903), [[github]](https://github.com/PetarV-/GAT)
    - [GAT-Based Predictor with DGL](../python/dgllife/model/model_zoo/gat_predictor.py)
    - [Example for Molecule Classification](property_prediction/moleculenet/classification.py)
- SchNet: A continuous-filter convolutional neural network for modeling quantum interactions [[paper]](https://arxiv.org/abs/1706.08566), [[github]](https://github.com/atomistic-machine-learning/SchNet)
    - [SchNet with DGL](../python/dgllife/model/model_zoo/schnet_predictor.py)
    - [Example for Molecule Regression](property_prediction/alchemy/main.py)
- Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective (MGCN) [[paper]](https://arxiv.org/abs/1906.11081)
    - [MGCN with DGL](../python/dgllife/model/model_zoo/mgcn_predictor.py)
    - [Example for Molecule Regression](property_prediction/alchemy/main.py)
- Neural Message Passing for Quantum Chemistry (MPNN) [[paper]](https://arxiv.org/abs/1704.01212), [[github]](https://github.com/brain-research/mpnn)
    - [MPNN with DGL](../python/dgllife/model/model_zoo/mpnn_predictor.py)
    - [Example for Molecule Regression](property_prediction/alchemy/main.py)
- Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism (AttentiveFP) [[paper]](https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.9b00959)
    - [AttentiveFP with DGL](../python/dgllife/model/model_zoo/attentivefp_predictor.py)
    - [Example for Molecule Regression](property_prediction/pubchem_aromaticity/main.py)
- Convolutional Networks on Graphs for Learning Molecular Fingerprints [[paper]](https://arxiv.org/abs/1509.09292)
    - [Neural Fingerprint with DGL](../python/dgllife/model/model_zoo/nf_predictor.py)

## Molecule Embeddings

- Strategies for Pre-training Graph Neural Networks [[paper]](https://arxiv.org/abs/1905.12265), [[github]](https://github.com/snap-stanford/pretrain-gnns)
    - [GIN with DGL](../python/dgllife/model/model_zoo/gin_predictor.py)
    - [Example for Computing Molecule Embeddings](molecule_embeddings/main.py)

## Generative Models

- Learning Deep Generative Models of Graphs (DGMG) [[paper]](https://arxiv.org/abs/1803.03324)
    - [DGMG with DGL](../python/dgllife/model/model_zoo/dgmg.py)
    - [Example Training Script](generative_models/dgmg)
- Junction Tree Variational Autoencoder for Molecular Graph Generation (JTNN) [[paper]](https://arxiv.org/abs/1802.04364)
    - [JTNN with DGL](../python/dgllife/model/model_zoo/jtnn)
    - [Example Training Script](generative_models/jtvae)

## Binding Affinity Prediction

- Atomic Convolutional Networks for Predicting Protein-Ligand Binding Affinity (ACNN) [[paper]](https://arxiv.org/abs/1703.10603), [[github]](https://github.com/deepchem/deepchem/tree/master/contrib/atomicconv)
    - [ACNN with DGL](../python/dgllife/model/model_zoo/acnn.py)
    - [Example Training Script](binding_affinity_prediction)
- PotentialNet for molecular property prediction (PotentialNet) [[paper]](https://pubs.acs.org/doi/10.1021/acscentsci.8b00507)
    - [PotentialNet with DGL](../python/dgllife/model/model_zoo/potentialnet.py)
    - [Example Training Script](binding_affinity_prediction)

## Reaction Prediction
- A graph-convolutional neural network model for the prediction of chemical reactivity [[paper]](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04228d#!divAbstract), [[github]](https://github.com/connorcoley/rexgen_direct)
    - An earlier version was published in NeurIPS 2017 as "Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network" [[paper]](https://arxiv.org/abs/1709.04555)
    - [WLN with DGL for Reaction Center Prediction](../python/dgllife/model/model_zoo/wln_reaction_center.py)
    - [Example Script](reaction_prediction/rexgen_direct)
