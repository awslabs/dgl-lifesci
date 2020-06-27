# Graph Property Prediction

GNN-based graph property prediction typically consists of the following steps:
1. Construct graphs
2. Prepare initial node (and edge) features for graphs
3. Use GNNs to update node representations of graphs
4. Compute graph representations out of updated node representations
5. Pass the graph representations to a MLP for final prediction

For those who are familiar with molecular fingerprints, GNN-based molecular representations 
can be viewed as learnable molecular fingerprints as first introduced in [1].

**To develop a GNN-based molecular property prediction model for your own dataset, see `csv_data_configuration`.**

## References

[1] Duvenaud et al. (2015) Convolutional networks on graphs for learning molecular fingerprints. *Advances in neural 
information processing systems (NeurIPS)*, 2224-2232.
