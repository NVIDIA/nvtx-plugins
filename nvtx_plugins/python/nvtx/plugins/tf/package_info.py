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

MAJOR = 0
MINOR = 1
PATCH = 2
PRE_RELEASE = ''

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'nvtx-plugins'
__contact_names__ = 'Ahmad Kiswani, Roni Forte, Jonathan Dekhtiar, Yaki Tebeka'
__contact_emails__ = 'akiswani@nvidia.com, rforte@nvidia.com, jdekhtiar@nvidia.com, ytebeka@nvidia.com'
__homepage__ = 'https://github.com/NVIDIA/nvtx-plugins/'
__repository_url__ = 'https://github.com/NVIDIA/nvtx-plugins/'
__download_url__ = 'https://github.com/NVIDIA/nvtx-plugins/'
__description__ = 'Python bindings for NVTX'
__license__ = 'Apache2'
__keywords__ = 'deep learning, machine learning, gpu, nvtx, nvidia, tensorflow, tf'
