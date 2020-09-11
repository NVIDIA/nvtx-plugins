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

import nvtx
from nvtx.plugins import DEFAULT_DOMAIN

# TODO(ahmadki): support category names
# TODO(ahmadki): move nvtx functionality to nvtx.plugins module ?


class BaseCallback(object):

    def __init__(self):
        self._range_stack = []

    def open_marker(self, message):
        marker_id = nvtx.start_range(message=message, color="blue", domain=DEFAULT_DOMAIN)
        self._range_stack.append(marker_id)

    def close_marker(self):
        marker_id = self._range_stack.pop()
        nvtx.end_range(marker_id, domain=DEFAULT_DOMAIN)
