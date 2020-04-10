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

from tensorflow.python.framework import load_library as _load_library
from tensorflow.python.platform import resource_loader

from nvtx.plugins.c_extensions_utils import TFExtension

from nvtx.common.ext_utils import get_extension_relative_path

__all__ = ["load_library"]


# Source:
# https://github.com/horovod/horovod/blob/abc3d/horovod/tensorflow/mpi_ops.py#L33
def load_library(extension):
    """Loads a .so file containing the specified operators.
    Args:
      extension: The name of the .so file to load.
    Raises:
      NotFoundError if were not able to load .so file.
    """

    if not isinstance(extension, TFExtension):
        raise ValueError(
            "The extension received is not an instance of `TFExtension`: %s" %
            extension
        )

    name = get_extension_relative_path(extension)
    name = os.path.join(*name.split(os.path.sep)[3:])

    filename = resource_loader.get_path_to_datafile(name)
    library = _load_library.load_op_library(filename)

    return library
