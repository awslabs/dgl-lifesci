#!/usr/bin/zsh
conda activate dgl
repeat 100 {python examples/binding_affinity_prediction/main.py -m PotentialNet -d PDBBind_refined_pocket_random >> hp_search_test2.txt}