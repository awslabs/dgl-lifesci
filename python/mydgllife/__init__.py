# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DGL-based package for applications in life science.

from .libinfo import __version__
from . import model

try:
    import rdkit
    from . import data
    from . import utils
except ImportError:
    print('RDKit is not installed, which is required for utils related to cheminformatics')
