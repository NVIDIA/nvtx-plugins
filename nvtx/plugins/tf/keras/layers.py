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

"""Keras layers.
"""

from tensorflow.keras.layers import Layer
from nvtx.plugins.tf.ops import nvtx_tf_ops


class NVTXStart(Layer):
    """An identity layer with a side effect of opening an NVTX marker.

    Note:
        The :func:`NVTXStart <NVTXStart>` and :func:`NVTXEnd <NVTXEnd>` layers
        must be used in pairs.

    Example:
        .. highlight:: python
        .. code-block:: python

            x, marker_id, domain_id = NVTXStart(message='Dense',
                                                domain_name='forward')(x)
            x = Dense(1024, activation='relu')(x)
            x = NVTXEnd(grad_message='Dense grad',
                        grad_domain_name='backwards')([x, marker_id, domain_id])

    Arguments:
        message: A ``string`` message to be associated with this layer.
        domain_name: An optional ``string`` domain name to be associated with
            this layer. If not provided the default NVTX domain will be used.
        trainable: ``bool``, if ``True`` will make this layer trainable.
            Used when this is the first layer in the graph to
            prevent an open ended marker during gradient calculation.
        name: An optional ``string`` name for the layer.

    Input shape:
        A ``Tensor`` object that is passed to ``output``.

    Output shape:
        ``list`` of length 3:
            - output: The inputs ``Tensor``.
            - marker_id: ``int64 Tensor``, sent to :func:`NVTXEnd <NVTXEnd>`.
            - domain_handle: ``int64 Tensor``. sent to :func:`NVTXEnd <NVTXEnd>`.

    """

    def __init__(self, message, domain_name=None,
                 trainable=False, **kwargs):
        super(NVTXStart, self).__init__(**kwargs)
        self.message = message
        self.domain_name = domain_name or ''
        self.trainable = trainable

    def build(self, input_shape):
        self.null_input = 1.
        if self.trainable:
            self.null_input = self.add_weight(name='null_input', shape=(),
                                              trainable=True, dtype='float32')
        super(NVTXStart, self).build(input_shape)

    def call(self, x):
        x, marker_id, domain_handle = nvtx_tf_ops.nvtx_start(inputs=x,
            message=self.message, domain_name=self.domain_name,
            null_input=self.null_input)
        return [x, marker_id, domain_handle]

    def compute_output_shape(self, input_shape):
        return [input_shape, (), ()]


class NVTXEnd(Layer):
    """An identity layer with a side effect of closing an NVTX marker.

    Note:
        The :func:`NVTXStart <NVTXStart>` and :func:`NVTXEnd <NVTXEnd>` layers
        must be used in pairs.

    Example:
        .. highlight:: python
        .. code-block:: python

            x, marker_id, domain_id = NVTXStart(message='Dense',
                                                domain_name='forward')(x)
            x = Dense(1024, activation='relu')(x)
            x = NVTXEnd(grad_message='Dense grad',
                        grad_domain_name='backwards')([x, marker_id, domain_id])

    Arguments:
        grad_message: An optional ``string`` message to be associated with
            the op gradient. If not provided an empty message will be used.
        grad_domain_name: An optional ``string`` domain name to be associated
            with this marker gradient. If not provided the default domain name
            will be used.
        name: An optional ``string`` name for the layer.

    Input shape:
        ``list`` of length 3:
            - inputs: The input ``Tensor``.
            - marker_id: ``int64 Tensor`` from :func:`NVTXStart <NVTXStart>`.
            - domain_handle: ``int64 Tensor`` from :func:`NVTXStart <NVTXStart>`.

    Output shape:
            A ``Tensor`` with ``inputs`` shape.

    """

    def __init__(self, grad_message=None, grad_domain_name=None, **kwargs):
        super(NVTXEnd, self).__init__(**kwargs)
        self.grad_message = grad_message or ''
        self.grad_domain_name = grad_domain_name or ''

    def build(self, input_shape):
        super(NVTXEnd, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list) and (len(x) == 3)
        inputs, marker_id, domain_handle = x
        output, _ = nvtx_tf_ops.nvtx_end(inputs=inputs, marker_id=marker_id,
                                         domain_handle=domain_handle,
                                         grad_message=self.grad_message,
                                         grad_domain_name=self.grad_domain_name)
        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [input_shape[0], ()]
