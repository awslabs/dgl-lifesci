#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 backend device"
}

if [ $# -ne 2 ]; then
    usage
    fail "Error: must specify backend and device"
fi

export DGLBACKEND=$1
export DGL_DOWNLOAD_DIR=${PWD}
export PYTHONPATH=${PWD}/python:$PYTHONPATH

conda activate ${DGLBACKEND}-ci

if [ $2 == "gpu" ]
then
  export CUDA_VISIBLE_DEVICES=0
  pip uninstall -y dgl-cu101
  pip install --pre dgl-cu101
else
  export CUDA_VISIBLE_DEVICES=-1
  pip uninstall -y dgl
  pip install --pre dgl
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
python -m pytest -v --junitxml=pytest_utils.xml tests/utils || fail "utils"