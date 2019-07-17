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

import tensorflow as tf
from nvtx.plugins.tf.base_callbacks import BaseCallback


class NVTXHook(BaseCallback, tf.train.SessionRunHook):
    """Hook that adds NVTX markers to a TensorFlow session session.
    
    Arguments:
        skip_n_steps: ``int``, skips adding markers for the first N
            ``session.run()`` calls.
        name: ``string``, a marker name for the session.

    """
    def __init__(self, skip_n_steps=0, name=None):
        super(NVTXHook, self).__init__()
        self.name = name
        self.step_counter = 0
        self.skip_n_steps = skip_n_steps
        self.iteration_message = 'step {iter}'

    def begin(self):
        self.step_counter = 0
        if self.name:
            self.open_marker(self.name)

    def before_run(self, run_context):
        if self.step_counter >= self.skip_n_steps:
            self.open_marker(self.iteration_message.format(iter=self.step_counter))

    def after_run(self, run_context, run_values):
        if self.step_counter >= self.skip_n_steps:
            self.close_marker(self.iteration_message.format(iter=self.step_counter))
        self.step_counter += 1

    def end(self, session):
        if self.name:
            self.close_marker(self.name)
