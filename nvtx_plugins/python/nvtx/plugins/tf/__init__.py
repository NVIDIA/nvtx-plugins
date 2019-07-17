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

"""NVTX Plugins"""

import os

from .package_info import __shortversion__
from .package_info import __version__

from .package_info import __package_name__
from .package_info import __contact_names__
from .package_info import __contact_emails__
from .package_info import __homepage__
from .package_info import __repository_url__
from .package_info import __download_url__
from .package_info import __description__
from .package_info import __license__
from .package_info import __keywords__

# Do not import during build of python package
if all(envname not in os.environ for envname in ("MAKEFLAGS", "MAKELEVEL")):
    import nvtx.plugins.tf.ops
    import nvtx.plugins.tf.estimator
    import nvtx.plugins.tf.keras
