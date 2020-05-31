#!/bin/bash

# Argument
#  - dev: cpu or gpu
if [ $# -ne 1 ]; then
    echo "Device argument required, can be cpu or gpu"
    exit -1
fi

dev=$1

set -e
. /opt/conda/etc/profile.d/conda.sh

rm -rf _deps
conda activate "pytorch-ci"
export PYTHONPATH=${PWD}/python:$PYTHONPATH
if [ "$dev" == "gpu" ]; then
  pip3 uninstall -y dgl
  pip3 install --pre dgl
  pushd python
  python3 setup.py install
else
  pip3 uninstall -y dgl-cu101
  pip3 install --pre dgl-cu101
  pushd python
  python3 setup.py install
fi
popd