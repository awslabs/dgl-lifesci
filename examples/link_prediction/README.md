# Link Prediction
Link prediction is a task to estimate the probability of links between nodes in a graph.

GNN-based link prediction typically consists of the following steps:
1. Construct graphs on biological networks
2. Prepare initial node (and edge) features for graphs
3. Use GNNs to update node representations of graphs
4. Compute the link representation from the product of its two updated nodes
5. Pass the link representations to a MLP for training and perform final link prediction

## Datasets
- **ogbl-ppa**: is an undirected, unweighted graph. Nodes represent proteins from 58 different species, 
and edges indicate biologically meaningful associations between proteins, e.g., physical interactions, 
co-expression, homology or genomic neighborhood. Each node contains a 58-dimensional one-hot feature 
vector that indicates the species that the corresponding protein comes from.[1]

## References

[1] Hu W, Fey M, Zitnik M, et al. Open graph benchmark: Datasets for machine learning on graphs[J]. 
arXiv preprint arXiv:2005.00687, 2020.
