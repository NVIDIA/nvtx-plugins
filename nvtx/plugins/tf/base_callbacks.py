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

# TODO(ahmadki): support domain names
# TODO(ahmadki): move nvtx functionality to nvtx.plugins module ?


class BaseCallback(object):
    def __init__(self):
        # TODO(ahmadki): try except OSError
        self.libnvtx = ctypes.cdll.LoadLibrary('libnvToolsExt.so')
        self.marker_ids = {}

    def open_marker(self, message):
        if self.marker_ids.get(message, None) is None:
            self.marker_ids[message] = []
        marker = self.libnvtx.nvtxRangeStartW(message)
        self.marker_ids[message].append(marker)

    def close_marker(self, message):
        if self.marker_ids.get(message, None) is not None:
            self.libnvtx.nvtxRangeEnd(self.marker_ids[message].pop())
            if len(self.marker_ids[message]) == 0:
                del self.marker_ids[message]
