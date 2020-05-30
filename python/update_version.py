# -*- coding: utf-8 -*-
#
# update_version.py
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
#
# This is the global script that set the version information of DGL-LifeSci.
# This script runs and update all the locations that related to versions
# List of affected files:
# - app-root/python/dgllife/__init__.py
# - app-root/conda/dgllife/meta.yaml

import os
import re

__version__ = "0.2.2"
print(__version__)

# Implementations
def update(file_name, pattern, repl):
    update = []
    hit_counter = 0
    need_update = False
    for l in open(file_name):
        result = re.findall(pattern, l)
        if result:
            assert len(result) == 1
            hit_counter += 1
            if result[0] != repl:
                l = re.sub(pattern, repl, l)
                need_update = True
                print("%s: %s->%s" % (file_name, result[0], repl))
            else:
                print("%s: version is already %s" % (file_name, repl))

        update.append(l)
    if hit_counter != 1:
        raise RuntimeError("Cannot find version in %s" % file_name)

    if need_update:
        with open(file_name, "w") as output_file:
            for l in update:
                output_file.write(l)

def main():
    curr_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_dir, ".."))
    # python path
    update(os.path.join(proj_root, "python/dgllife/libinfo.py"),
           r"(?<=__version__ = \")[.0-9a-z]+", __version__)
    # conda
    update(os.path.join(proj_root, "conda/dgllife/meta.yaml"),
           "(?<=version: \")[.0-9a-z]+", __version__)

if __name__ == '__main__':
    main()
