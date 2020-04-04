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

"""Keras callback.
"""

import tensorflow as tf
from nvtx.plugins.tf.base_callbacks import BaseCallback


class NVTXCallback(BaseCallback, tf.keras.callbacks.Callback):
    """Callback that adds NVTX markers to a keras session.
    """

    def __init__(self, **kwargs):
        super(NVTXCallback, self).__init__(**kwargs)
        self.epoch_message = 'epoch {epoch}'
        self.batch_message = 'batch {batch}'

    def on_epoch_begin(self, epoch, logs=None):
        self.open_marker(self.epoch_message.format(epoch=epoch))

    def on_epoch_end(self, epoch, logs=None):
        self.close_marker(self.epoch_message.format(epoch=epoch))

    def on_train_batch_begin(self, batch, logs=None):
        self.open_marker(self.batch_message.format(batch=batch))

    def on_train_batch_end(self, batch, logs=None):
        self.close_marker(self.batch_message.format(batch=batch))

    def on_test_batch_begin(self, batch, logs=None):
        self.open_marker(self.batch_message.format(batch=batch))

    def on_test_batch_end(self, batch, logs=None):
        self.close_marker(self.batch_message.format(batch=batch))

    def on_predict_batch_begin(self, batch, logs=None):
        self.open_marker(self.batch_message.format(batch=batch))

    def on_predict_batch_end(self, batch, logs=None):
        self.close_marker(self.batch_message.format(batch=batch))

    def on_train_begin(self, logs=None):
        self.open_marker('Train')

    def on_train_end(self, logs=None):
        self.close_marker('Train')

    def on_test_begin(self, logs=None):
        self.open_marker('Test')

    def on_test_end(self, logs=None):
        self.close_marker('Test')

    def on_predict_begin(self, logs=None):
        self.open_marker('Predict')

    def on_predict_end(self, logs=None):
        self.close_marker('Predict')

