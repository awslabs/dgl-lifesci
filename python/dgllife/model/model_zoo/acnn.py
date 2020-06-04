# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Atomic Convolutional Networks for Predicting Protein-Ligand Binding Affinity"""
# pylint: disable=C0103, C0123, W0221, E1101, R1721

import itertools
import numpy as np
import torch
import torch.nn as nn

from dgl import BatchedDGLHeteroGraph
from dgl.nn.pytorch import AtomicConv

__all__ = ['ACNN']

def truncated_normal_(tensor, mean=0., std=1.):
    """Fills the given tensor in-place with elements sampled from the truncated normal
    distribution parameterized by mean and std.

    The generated values follow a normal distribution with specified mean and
    standard deviation, except that values whose magnitude is more than 2 std
    from the mean are dropped.

    We credit to Ruotian Luo for this implementation:
    https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15.

    Parameters
    ----------
    tensor : Float32 tensor of arbitrary shape
        Tensor to be filled.
    mean : float
        Mean of the truncated normal distribution.
    std : float
        Standard deviation of the truncated normal distribution.
    """
    shape = tensor.shape
    tmp = tensor.new_empty(shape + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class ACNNPredictor(nn.Module):
    """Predictor for ACNN.

    Parameters
    ----------
    in_size : int
        Number of radial filters used.
    hidden_sizes : list of int
        Specifying the hidden sizes for all layers in the predictor.
    weight_init_stddevs : list of float
        Specifying the standard deviations to use for truncated normal
        distributions in initialzing weights for the predictor.
    dropouts : list of float
        Specifying the dropouts to use for all layers in the predictor.
    features_to_use : None or float tensor of shape (T)
        In the original paper, these are atomic numbers to consider, representing the types
        of atoms. T for the number of types of atomic numbers. Default to None.
    num_tasks : int
        Output size.
    """
    def __init__(self, in_size, hidden_sizes, weight_init_stddevs,
                 dropouts, features_to_use, num_tasks):
        super(ACNNPredictor, self).__init__()

        if type(features_to_use) != type(None):
            in_size *= len(features_to_use)

        modules = []
        for i, h in enumerate(hidden_sizes):
            linear_layer = nn.Linear(in_size, h)
            truncated_normal_(linear_layer.weight, std=weight_init_stddevs[i])
            modules.append(linear_layer)
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropouts[i]))
            in_size = h
        linear_layer = nn.Linear(in_size, num_tasks)
        truncated_normal_(linear_layer.weight, std=weight_init_stddevs[-1])
        modules.append(linear_layer)
        self.project = nn.Sequential(*modules)

    def forward(self, batch_size, frag1_node_indices_in_complex, frag2_node_indices_in_complex,
                ligand_conv_out, protein_conv_out, complex_conv_out):
        """Perform the prediction.

        Parameters
        ----------
        batch_size : int
            Number of datapoints in a batch.
        frag1_node_indices_in_complex : Int64 tensor of shape (V1)
            Indices for atoms in the first fragment (protein) in the batched complex.
        frag2_node_indices_in_complex : list of int of length V2
            Indices for atoms in the second fragment (ligand) in the batched complex.
        ligand_conv_out : Float32 tensor of shape (V2, K * T)
            Updated ligand node representations. V2 for the number of atoms in the
            ligand, K for the number of radial filters, and T for the number of types
            of atomic numbers.
        protein_conv_out : Float32 tensor of shape (V1, K * T)
            Updated protein node representations. V1 for the number of
            atoms in the protein, K for the number of radial filters,
            and T for the number of types of atomic numbers.
        complex_conv_out : Float32 tensor of shape (V1 + V2, K * T)
            Updated complex node representations. V1 and V2 separately
            for the number of atoms in the ligand and protein, K for
            the number of radial filters, and T for the number of
            types of atomic numbers.

        Returns
        -------
        Float32 tensor of shape (B, O)
            Predicted protein-ligand binding affinity. B for the number
            of protein-ligand pairs in the batch and O for the number of tasks.
        """
        ligand_feats = self.project(ligand_conv_out)   # (V1, O)
        protein_feats = self.project(protein_conv_out) # (V2, O)
        complex_feats = self.project(complex_conv_out) # (V1+V2, O)

        ligand_energy = ligand_feats.reshape(batch_size, -1).sum(-1, keepdim=True)   # (B, O)
        protein_energy = protein_feats.reshape(batch_size, -1).sum(-1, keepdim=True) # (B, O)

        complex_ligand_energy = complex_feats[frag1_node_indices_in_complex].reshape(
            batch_size, -1).sum(-1, keepdim=True)
        complex_protein_energy = complex_feats[frag2_node_indices_in_complex].reshape(
            batch_size, -1).sum(-1, keepdim=True)
        complex_energy = complex_ligand_energy + complex_protein_energy

        return complex_energy - (ligand_energy + protein_energy)

class ACNN(nn.Module):
    """Atomic Convolutional Networks.

    The model was proposed in `Atomic Convolutional Networks for
    Predicting Protein-Ligand Binding Affinity <https://arxiv.org/abs/1703.10603>`__.

    The prediction proceeds as follows:

    1. Perform message passing to update atom representations for the
       ligand, protein and protein-ligand complex.
    2. Predict the energy of atoms from their representations with an MLP.
    3. Take the sum of predicted energy of atoms within each molecule for
       predicted energy of the ligand, protein and protein-ligand complex.
    4. Make the final prediction by subtracting the predicted ligand and protein
       energy from the predicted complex energy.

    Parameters
    ----------
    hidden_sizes : list of int
        ``hidden_sizes[i]`` gives the size of hidden representations in the i-th
        hidden layer of the MLP. By Default, ``[32, 32, 16]`` will be used.
    weight_init_stddevs : list of float
        ``weight_init_stddevs[i]`` gives the std to initialize parameters in the
        i-th layer of the MLP. Note that ``len(weight_init_stddevs) == len(hidden_sizes) + 1``
        due to the output layer. By default, we use ``1 / sqrt(hidden_sizes[i])`` for hidden
        layers and 0.01 for the output layer.
    dropouts : list of float
        ``dropouts[i]`` gives the dropout in the i-th hidden layer of the MLP. By default,
        no dropout is used.
    features_to_use : None or float tensor of shape (T)
        In the original paper, these are atomic numbers to consider, representing the types
        of atoms. T for the number of types of atomic numbers. If None, we use same parameters
        for all atoms regardless of their type. Default to None.
    radial : list
        The list consists of 3 sublists of floats, separately for the
        options of interaction cutoff, the options of rbf kernel mean and the
        options of rbf kernel scaling. By default,
        ``[[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]]`` will be used.
    num_tasks : int
        Number of output tasks. Default to 1.
    """
    def __init__(self, hidden_sizes=None, weight_init_stddevs=None, dropouts=None,
                 features_to_use=None, radial=None, num_tasks=1):
        super(ACNN, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 32, 16]

        if weight_init_stddevs is None:
            weight_init_stddevs = [1. / float(np.sqrt(hidden_sizes[i]))
                                   for i in range(len(hidden_sizes))]
            weight_init_stddevs.append(0.01)

        if dropouts is None:
            dropouts = [0. for _ in range(len(hidden_sizes))]

        if radial is None:
            radial = [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]]
        # Take the product of sets of options and get a list of 3-tuples.
        radial_params = [x for x in itertools.product(*radial)]
        radial_params = torch.stack(list(map(torch.tensor, zip(*radial_params))), dim=1)

        interaction_cutoffs = radial_params[:, 0]
        rbf_kernel_means = radial_params[:, 1]
        rbf_kernel_scaling = radial_params[:, 2]

        self.ligand_conv = AtomicConv(interaction_cutoffs, rbf_kernel_means,
                                      rbf_kernel_scaling, features_to_use)
        self.protein_conv = AtomicConv(interaction_cutoffs, rbf_kernel_means,
                                       rbf_kernel_scaling, features_to_use)
        self.complex_conv = AtomicConv(interaction_cutoffs, rbf_kernel_means,
                                       rbf_kernel_scaling, features_to_use)
        self.predictor = ACNNPredictor(radial_params.shape[0], hidden_sizes,
                                       weight_init_stddevs, dropouts, features_to_use, num_tasks)

    def forward(self, graph):
        """Apply the model for prediction.

        Parameters
        ----------
        graph : DGLHeteroGraph
            DGLHeteroGraph consisting of the ligand graph, the protein graph
            and the complex graph, along with preprocessed features. For a batch of
            protein-ligand pairs, we assume zero padding is performed so that the
            number of ligand and protein atoms is the same in all pairs.

        Returns
        -------
        Float32 tensor of shape (B, O)
            Predicted protein-ligand binding affinity. B for the number
            of protein-ligand pairs in the batch and O for the number of tasks.
        """
        ligand_graph = graph[('ligand_atom', 'ligand', 'ligand_atom')]
        ligand_graph_node_feats = ligand_graph.ndata['atomic_number']
        assert ligand_graph_node_feats.shape[-1] == 1
        ligand_graph_distances = ligand_graph.edata['distance']
        ligand_conv_out = self.ligand_conv(ligand_graph,
                                           ligand_graph_node_feats,
                                           ligand_graph_distances)

        protein_graph = graph[('protein_atom', 'protein', 'protein_atom')]
        protein_graph_node_feats = protein_graph.ndata['atomic_number']
        assert protein_graph_node_feats.shape[-1] == 1
        protein_graph_distances = protein_graph.edata['distance']
        protein_conv_out = self.protein_conv(protein_graph,
                                             protein_graph_node_feats,
                                             protein_graph_distances)

        complex_graph = graph[:, 'complex', :]
        complex_graph_node_feats = complex_graph.ndata['atomic_number']
        assert complex_graph_node_feats.shape[-1] == 1
        complex_graph_distances = complex_graph.edata['distance']
        complex_conv_out = self.complex_conv(complex_graph,
                                             complex_graph_node_feats,
                                             complex_graph_distances)

        frag1_node_indices_in_complex = torch.where(complex_graph.ndata['_TYPE'] == 0)[0]
        frag2_node_indices_in_complex = list(set(range(complex_graph.number_of_nodes())) -
                                             set(frag1_node_indices_in_complex.tolist()))

        # Hack the case when we are working with a single graph.
        if not isinstance(graph, BatchedDGLHeteroGraph):
            graph.batch_size = 1

        return self.predictor(
            graph.batch_size,
            frag1_node_indices_in_complex,
            frag2_node_indices_in_complex,
            ligand_conv_out, protein_conv_out, complex_conv_out)
