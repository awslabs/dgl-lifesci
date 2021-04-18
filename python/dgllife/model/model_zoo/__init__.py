# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Collection of model architectures

# Molecular property prediction
from .attentivefp_predictor import *
from .gat_predictor import *
from .gcn_predictor import *
from .gin_predictor import *
from .nf_predictor import *
from .mgcn_predictor import *
from .mlp_predictor import *
from .mpnn_predictor import *
from .schnet_predictor import *
from .weave_predictor import *
from .gnn_ogb_predictor import *
from .pagtn_predictor import *

# Generative models
try:
    # DGMG requires RDKit support
    from .dgmg import *
    from .jtvae import *
except ImportError:
    pass

# Reaction prediction
from .wln_reaction_center import *
from .wln_reaction_ranking import *

# Protein-Ligand Binding
from .acnn import *

# Link prediction
from .hadamard_link_predictor import *
