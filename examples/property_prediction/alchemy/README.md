# Alchemy

## Dataset Introduction

[1] introduced a dataset comprised of 12 quantum mechanical properties of 119,487 organic molecules with up to 14 
heavy atoms, sampled from the GDB MedChem database. 

## Modeling

### Pre-processing

[1] performed a stratified split of the dataset so that each of the training, validation and test set covers the 
full range of provided labels. By default, we construct a complete graph for each molecule, i.e. each pair of atoms  
is connected. The details for node/edge featurization can be found 
[here](https://lifesci.dgl.ai/api/data.html#alchemy-for-quantum-chemistry).

### Training and Evaluation

We support three models for this dataset -- SchNet [2], MPNN [3], MGCN [4]. To train from scratch, run 

```bash
python main.py -m X
```

where `X` can be `SchNet`, `MPNN`, or `MGCN`.

## References

[1] Chen et al. (2019) Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models.
[2] Schütt et al. (2017) SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. 
*Advances in Neural Information Processing Systems (NeurIPS)*, 992-1002.
[3] Gilmer et al. (2017) Neural Message Passing for Quantum Chemistry. *Proceedings of the 34th International Conference on 
Machine Learning*, JMLR. 1263-1272.
[4] Lu et al. (2019) Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective. 
*The 33rd AAAI Conference on Artificial Intelligence*. 
