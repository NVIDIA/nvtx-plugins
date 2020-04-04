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

try:
    from nvtx.c_extensions_utils import CustomExtension
    from nvtx.c_extensions_utils import TFExtension

except ImportError:
    import imp
    c_extensions_utils = imp.load_source(
        'c_extensions_utils',
        'nvtx_plugins/python/nvtx/c_extensions_utils.py')

    from c_extensions_utils import CustomExtension
    from c_extensions_utils import TFExtension

__all__ = [
    "collect_all_c_extensions"
]


def collect_all_c_extensions():
    return [x for x in CustomExtension.get_instances()]


tensorflow_nvtx_lib = TFExtension(
    name='nvtx.plugins.tf.lib.nvtx_ops',
    ###############
    define_macros=[('GOOGLE_CUDA', '1')],
    depends=[],
    export_symbols=[],
    extra_compile_args=[],
    extra_link_args=[],
    include_dirs=[],
    language=None,
    libraries=[],
    library_dirs=[],
    py_limited_api=False,
    runtime_library_dirs=[],
    sources=[
        'nvtx_plugins/cc/common/nvtx_custom_markers.cc',
        'nvtx_plugins/cc/tensorflow/nvtx_ops.cc',
        'nvtx_plugins/cc/tensorflow/nvtx_kernels.cc',
    ],
    swig_opts=[],
    undef_macros=["NDEBUG"],
)
