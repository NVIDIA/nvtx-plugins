# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Code Inspired by: https://github.com/horovod/horovod/blob/master/setup.py
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
#
# Modified by NVIDIA to fit our requirements
#
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

import sys
import weakref

from setuptools import Extension

__all__ = [
    "CustomExtension",
    "PyTExtension",
    "TFExtension",
]


def _is_framework_imported(framework):
    if framework not in ["tensorflow", "torch"]:
        raise ValueError("Unsupported framework requested: %s" % framework)

    return framework in sys.modules


class CustomExtension(Extension):

    _instances = set()

    def __init__(self, name, sources, *args, **kwargs):
        super(CustomExtension, self).__init__(name=name, sources=sources, *args, **kwargs)

        self._instances.add(weakref.ref(self))
        self._built_with_success = False

    @classmethod
    def get_instances(cls):
        dead = set()

        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj

            else:
                dead.add(ref)

        cls._instances -= dead


class PyTExtension(CustomExtension):

    def __init__(self, name, sources, *args, **kwargs):

        super(PyTExtension, self).__init__(name=name, sources=sources, *args, **kwargs)

        try:
            from torch.utils.cpp_extension import CUDAExtension
            torch_ext = CUDAExtension(name=name, sources=sources, *args, **kwargs)

            # Patch the existing extension object.
            for k, v in torch_ext.__dict__.items():
                self.__dict__[k] = v
        except ModuleNotFoundError:
            pass

        self.optional = not _is_framework_imported(framework="torch")


class TFExtension(CustomExtension):

    def __init__(self, name, sources, *args, **kwargs):

        super(TFExtension, self).__init__(name=name, sources=sources, *args, **kwargs)

        self.optional = not _is_framework_imported(framework="tensorflow")