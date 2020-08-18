# Junction Tree Variational Autoencoder for Molecular Graph Generation (JTNN)

Wengong Jin, Regina Barzilay, Tommi Jaakkola. 
Junction Tree Variational Autoencoder for Molecular Graph Generation. 
*arXiv preprint arXiv:1802.04364*, 2018.

JTNN uses algorithm called junction tree algorithm to form a tree from the molecular graph. 
Then the model will encode the tree and graph into two separate vectors `z_G` and `z_T`. Details can
be found in original paper. The brief process is as below (from original paper): 

![image](https://user-images.githubusercontent.com/8686776/63677300-3fb6d980-c81f-11e9-8a65-57c8b03aaf52.png)

**Goal**: JTNN is an auto-encoder model, aiming to learn hidden representation for molecular graphs. 
These representations can be used for downstream tasks, such as property prediction, or molecule optimizations.

## Dataset

### ZINC

> The ZINC database is a curated collection of commercially available chemical compounds 
prepared especially for virtual screening. (introduction from Wikipedia)

Generally speaking, molecules in the ZINC dataset are more drug-like. We use ~220,000 
molecules for training and 5000 molecules for validation. 

### GuacaMol

> The dataset are used by a series of baseline model implementations for the [guacamol](https://github.com/BenevolentAI/guacamol) benchmark for training generative models. The datasets are derived from the ChEMBL 24 database.

Generally speaking, the advantage of ChEMBL is that it contains only molecules that have been synthesized and
tested against biological targets. Other datasets, such as ZINC, contain, at least to some
degree, virtual molecules that are likely to be synthesizable, but have not been made yet. (introduction from their [paper](https://arxiv.org/abs/1811.09621)) 
We use ~1,270,000 molecules for training and ~50,000 molecules for validation.

### Download
For ZINC dataset, you can run with
```
python train.py
```
The dataset will be downloaded automatically. 

For GuacaMol dataset, you can download the training-set and validation-set with 
```
mkdir -p $FILENAME
wget https://ndownloader.figshare.com/files/13612760 -O $FILENAME/guacamol_v1_train.smiles
wget https://ndownloader.figshare.com/files/13612766 -O $FILENAME/guacamol_v1_valid.smiles
``` 
`$FILENAME` specifies the filename to save the dataset.


### Preprocessing

Class `JTNNDataset` will process a SMILES string into a dict, consisting of a junction tree, a graph with 
encoded nodes(atoms) and edges(bonds), and other information for model to use.

## Usage

### Training
####ZINC
For ZINC dataset, to start training, use `python train.py`. By default, the script will use ZINC dataset
 with preprocessed vocabulary, and save model checkpoint periodically in the current working directory. 
 
####GuacaMol
For GuacaMol dataset, you need to process the training dataset to get vocabulary first with
```
python vocab.py -d $DATASET -v  $VOCABULARY
```
`$DATASET` specifies the filename of the training-set.  
`$VOCABULARY` specifies the filename to save the processed vocabulary.  
  
To start training on GuacaMol, use
```
python train.py -t $DATASET -v  $VOCABULARY -s $CHECKPOINTS
```
`$DATASET` specifies the filename of the training-set.  
`$VOCABULARY` specifies the filename of processed vocabulary.    
`$CHECKPOINTS` specifies the filename to save model checkpoint periodically. 
### Evaluation

To start evaluation, use `python reconstruct_eval.py`. By default, we will perform evaluation with 
DGL's pre-trained model. During the evaluation, the program will print out the success rate of 
molecule reconstruction.

### Pre-trained models

Below gives the statistics of our pre-trained `JTNN_ZINC` model. 

| Pre-trained model  | % Reconstruction Accuracy
| ------------------ | -------
| `JTNN_ZINC`        |  73.7             

### Visualization

Here we draw some "neighbor" of a given molecule, by adding noises on the intermediate representations. 
You can download the script with `wget https://data.dgl.ai/dgllife/jtnn_viz_neighbor_mol.ipynb`. 
Please put this script at the current directory (`examples/pytorch/model_zoo/chem/generative_models/jtnn/`).

#### Given Molecule
![image](https://user-images.githubusercontent.com/8686776/63773593-0d37da00-c90e-11e9-8933-0abca4b430db.png)
#### Neighbor Molecules
![image](https://user-images.githubusercontent.com/8686776/63773602-1163f780-c90e-11e9-8341-5122dc0d0c82.png)

### Dataset configuration

If you want to use your own dataset, please create a file with one SMILES a line as below

```
CCO
Fc1ccccc1
```

You can generate the vocabulary file corresponding to your dataset with `python vocab.py -d X -v Y`, where `X` 
is the path to the dataset and `Y` is the path to the vocabulary file to save. An example vocabulary file 
corresponding to the two molecules above will be

```
CC
CF
C1=CC=CC=C1
CO
```

If you want to develop a model based on DGL's pre-trained model, it's important to make sure that the vocabulary 
generated above is a subset of the vocabulary we use for the pre-trained model. By running `vocab.py` above, we 
also check if the new vocabulary is a subset of the vocabulary we use for the pre-trained model and print the 
result in the terminal as follows:

```
The new vocabulary is a subset of the default vocabulary: True
```

To train on this new dataset, run

```
python train.py -t X
```

where `X` is the path to the new dataset. If you want to use the vocabulary generated above, also add `-v Y`, where 
`Y` is the path to the vocabulary file we just saved.

To evaluate on this new dataset, run `python reconstruct_eval.py` with arguments same as above.
