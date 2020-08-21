# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from joblib import Parallel, delayed, cpu_count

from .analysis import *
from .complex_to_graph import *
from .early_stop import *
from .eval import *
from .featurizers import *
from .io import *
from .mol_to_graph import *
from .splitters import *


def pmap(pickleable_fn, data, n_jobs=cpu_count() - 1, verbose=1, **kwargs):
    """
    Parallel map using joblib.

    :param pickleable_fn: Fn to map over data.
    :param data: Data to be mapped over.
    :param n_jobs: CPU parallelism, uses 1 less than number detected by default.
    :param verbose: Job logging verbosity, set to 0 to silence.
    :param kwargs: Additional args for f
    :return: Mapped output.
    """
    return Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(pickleable_fn)(d, **kwargs) for d in data
    )
