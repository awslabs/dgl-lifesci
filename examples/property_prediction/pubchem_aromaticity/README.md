# Pubchem Aromaticity

## Dataset Introduction

[1] extracted a total of 3945 molecules with 0-40 aromatic atoms from the PubChem BioAssay dataset for predicting 
the number of aromatic atoms of molecules.

## Modeling

### Pre-processing

We randomly split the dataset into training/validation/test subsets with a ratio of 0.8:0.1:0.1. 
For featurization, we exclude all bond features and all atom aromatic features as in [1]. 

### Training and Evaluation

To train from scratch, run 

```bash
python main.py
```

You can also directly evaluate a pre-trained model with

```bash
python main.py -p
```

which will yield a test RMSE of 0.7508.

## Visualization

In computing molecular representations out of atom representations, the model learns 
to assign some weights to the atoms, which can be viewed as the importance of atoms. 
[1] visualizes the weights of the atoms for possible interpretations like the figure below. 
We provide a jupyter notebook for performing the visualization and you can download it with 
`wget https://data.dgl.ai/dgllife/attentive_fp/atom_weight_visualization.ipynb`.

![](https://data.dgl.ai/dgllife/attentive_fp_vis_example.png)

## References

[1] Xiong et al. (2019) Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph 
Attention Mechanism. *Journal of Medicinal Chemistry*.
