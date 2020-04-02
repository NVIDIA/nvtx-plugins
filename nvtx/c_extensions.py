# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from nvtx.c_extensions_utils import PyTExtension
from nvtx.c_extensions_utils import TFExtension

__all__ = [
    "tensorflow_nvtx_lib"
]

tensorflow_nvtx_lib = TFExtension(
    name='nvtx.plugins.tf.lib.nvtx_ops',
    ###############
    define_macros=[('GOOGLE_CUDA', '1')],
    depends=[],
    export_symbols=[],
    extra_compile_args=['-lnvToolsExt'],
    extra_link_args=['-lnvToolsExt'],
    include_dirs=[],
    language=None,
    libraries=[],
    library_dirs=[],
    py_limited_api=False,
    runtime_library_dirs=[],
    sources=[
        'nvtx/cc/common/nvtx_custom_markers.cc',
        'nvtx/cc/tensorflow/nvtx_ops.cc',
        'nvtx/cc/tensorflow/nvtx_kernels.cc',
    ],
    swig_opts=[],
    undef_macros=["NDEBUG"],
)
