# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
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
# =============================================================================
#
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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
# ==============================================================================

import os
import sysconfig

from tensorflow.python.framework import load_library as _load_library
from tensorflow.python.platform import resource_loader

__all__ = ["get_ext_suffix", "load_library"]


# Source: https://github.com/horovod/horovod/blob/abc3d88544/horovod/tensorflow/mpi_ops.py#L33
def load_library(name):
    """Loads a .so file containing the specified operators.
    Args:
      name: The name of the .so file to load.
    Raises:
      NotFoundError if were not able to load .so file.
    """

    filename = resource_loader.get_path_to_datafile(name)
    library = _load_library.load_op_library(filename)
    return library


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def get_extension_full_path(pkg_path, *args):
    assert len(args) >= 1
    dir_path = os.path.join(os.path.dirname(pkg_path), *args[:-1])
    full_path = os.path.join(dir_path, args[-1] + get_ext_suffix())
    return full_path


def check_extension(ext_name, ext_env_var, pkg_path, *args):
    full_path = get_extension_full_path(pkg_path, *args)

    if not os.path.exists(full_path):
        raise ImportError(
            'NVTX-Plugins %s has not been built.  If this is not expected, please reinstall to debug the build error' %
            ext_name
        )