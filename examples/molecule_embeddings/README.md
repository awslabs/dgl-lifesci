# Pre-trained Molecule Embeddings

We can use pre-trained GNNs to compute molecule embeddings. In particular, 
[Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265) develops multiple 
approaches for pre-training GNNs on two million molecules. In addition to training the models for 
molecular property prediction with supervised learning, we can also combine with the following 
semi-supervised learning approaches:

- **contextpred**: Use subgraphs of molecular graphs for predicting surrounding graph structures.
- **infomax**: Train the models to maximize the mutual information between local node representations 
and a global graph representation.
- **edgepred**: Encourage nearby nodes to have similar representations, while enforcing that the 
representations of disparate nodes are highly distinct.
- **masking**: Randomly mask node and edge attributes and let the model predict those attributes.

We adapted the trained models they released for computing molecule embeddings, 
which can be used like traditional molecular fingerprints.

## Data Preparation

You can either prepare a `.txt` file where each line corresponds to the SMILES string for a molecule or 
a `.csv` file where a column contains the SMILES strings for all molecules.

## Usage

To compute molecule embeddings by providing a `.txt` file, do 

```bash
python main.py -fi A -m B
```

To compute molecule embeddings by providing a `.csv` file, do

```bash
python main.py -fi A -m B -fo csv -sc C
```

where:
- `A` specifies the path to the `.txt` file or the `.csv` file
- `B` specifies the pre-trained model to use, which can be `gin_supervised_contextpred`, 
`gin_supervised_infomax`, `gin_supervised_edgepred`, `gin_supervised_masking`.
- `C` specifies the header for SMILES column in the `.csv` file

Other optional arguments include:
- **batch_size**: `-b D` can be used to specify the batch size for computation. 
By default we use `256`.
- **out_dir**: `-o E` can be used to specify the directory for storing the computation results. 
By default we use `results`.

## Results

We store two files in the output directory:
- `mol_parsed.npy`: Since we may not be able to parse some SMILES strings with RDKit, we use a 
bool numpy array `a` where `a[i]` is True if the i-th SMILES string can be parsed by RDKit.
- `mol_emb.npy`: We use a numpy array `b` for storing the computed molecule embeddings where `b[i]` 
gives the molecule embedding of the i-th molecule which can be parsed by RDKit.
