# -*- coding: utf-8 -*-
#
# test_early_stop.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.nn as nn

from dgllife.utils import EarlyStopping

def remove_file(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass

def test_early_stopping_high():
    model1 = nn.Linear(2, 3)
    stopper = EarlyStopping(mode='higher',
                            patience=1,
                            filename='test.pkl')

    # Save model in the first step
    stopper.step(1., model1)
    model1.weight.data = model1.weight.data + 1
    model2 = nn.Linear(2, 3)
    stopper.load_checkpoint(model2)
    assert not torch.allclose(model1.weight, model2.weight)

    # Save model checkpoint with performance improvement
    model1.weight.data = model1.weight.data + 1
    stopper.step(2., model1)
    stopper.load_checkpoint(model2)
    assert torch.allclose(model1.weight, model2.weight)

    # Stop when no improvement observed
    model1.weight.data = model1.weight.data + 1
    assert stopper.step(0.5, model1)
    stopper.load_checkpoint(model2)
    assert not torch.allclose(model1.weight, model2.weight)

    remove_file('test.pkl')

def test_early_stopping_low():
    model1 = nn.Linear(2, 3)
    stopper = EarlyStopping(mode='lower',
                            patience=1,
                            filename='test.pkl')

    # Save model in the first step
    stopper.step(1., model1)
    model1.weight.data = model1.weight.data + 1
    model2 = nn.Linear(2, 3)
    stopper.load_checkpoint(model2)
    assert not torch.allclose(model1.weight, model2.weight)

    # Save model checkpoint with performance improvement
    model1.weight.data = model1.weight.data + 1
    stopper.step(0.5, model1)
    stopper.load_checkpoint(model2)
    assert torch.allclose(model1.weight, model2.weight)

    # Stop when no improvement observed
    model1.weight.data = model1.weight.data + 1
    assert stopper.step(2, model1)
    stopper.load_checkpoint(model2)
    assert not torch.allclose(model1.weight, model2.weight)

    remove_file('test.pkl')

if __name__ == '__main__':
    test_early_stopping_high()
    test_early_stopping_low()
