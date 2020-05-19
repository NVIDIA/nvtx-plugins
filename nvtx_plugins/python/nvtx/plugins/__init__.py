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

import ctypes
nvtx_clib = ctypes.cdll.LoadLibrary('libnvToolsExt.so')

from nvtx.plugins.package_info import __shortversion__
from nvtx.plugins.package_info import __version__

from nvtx.plugins.package_info import __package_name__
from nvtx.plugins.package_info import __contact_names__
from nvtx.plugins.package_info import __contact_emails__
from nvtx.plugins.package_info import __homepage__
from nvtx.plugins.package_info import __repository_url__
from nvtx.plugins.package_info import __download_url__
from nvtx.plugins.package_info import __description__
from nvtx.plugins.package_info import __license__
from nvtx.plugins.package_info import __keywords__

from nvtx.plugins.common.logger import Logger as _Logger
logging = _Logger()

from nvtx.plugins.common.decorators import deprecated
from nvtx.plugins.common.decorators import deprecated_argument
from nvtx.plugins.common.decorators import deprecated_alias


@deprecated(instructions='useless function, please remove')
def lol():
    pass


@deprecated(instructions='useless class, please remove')
class LOLClass(object):
    pass


class OKClass(object):
    @deprecated(instructions='useless method, please remove')
    def lolmethod(self):
        pass


@deprecated_argument(deprecated_args=["arg1"])
def lolfunc():
    pass


@deprecated_alias(deprecated_aliases={"old": "new"})
def lolfunc_alias(new):
    print("new argument:", new)
