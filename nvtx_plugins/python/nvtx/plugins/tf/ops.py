# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import wrapt
import tensorflow as tf

from tensorflow.python.framework import ops

from nvtx.plugins.tf.ext_utils import load_library
from nvtx.plugins.tf.ext_utils import get_ext_suffix

__all__ = ['nvtx_tf_ops', 'start', 'end', 'trace']


nvtx_tf_ops = load_library('lib/nvtx_ops' + get_ext_suffix())


@ops.RegisterGradient('NvtxStart')
def _nvtx_start_grad(op, grad, marker_id, domain_handle):
    # grad_message and grad_domain_name are not used
    if not isinstance(marker_id, tf.Tensor) and marker_id is None:
        raise RuntimeError('Error in nvtx range %s. '
                           'Make sure all nvtx ranges are closed' % op.name)

    grad, null_grad = nvtx_tf_ops.nvtx_end(inputs=grad,
        marker_id=marker_id, domain_handle=domain_handle,
        grad_message=op.inputs[2], grad_domain_name=op.inputs[3])
    return [grad, null_grad, None, None]


@ops.RegisterGradient('NvtxEnd')
def _nvtx_end_grad(op, grad, null_grad):
    grad, marker_id, domain_handle = nvtx_tf_ops.nvtx_start(
        inputs=grad, null_input=1.,
        message=op.inputs[3], domain_name=op.inputs[4])
    return [grad, marker_id, domain_handle, None, None]


def start(inputs, message, domain_name=None,
          grad_message=None, grad_domain_name=None,
          trainable=False, enabled=True, name=None):
    """An identity operation with a side effect of opening an NVTX marker.

    Note:
        The :func:`ops.start <start>` and :func:`ops.end <end>` operations
        must be used in pairs.

    Example:
        .. highlight:: python
        .. code-block:: python

            x, nvtx_context = nvtx.plugins.tf.ops.start(x, message='Dense 1-3',
                domain_name='Forward', grad_domain_name='Gradient')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_1')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_2')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_3')
            x = nvtx.plugins.tf.ops.end(x, nvtx_context)

    Arguments:
        inputs: A ``Tensor`` object that is passed to ``output``.
        message: A ``string`` message to be associated with this marker.
        domain_name: An optional ``string`` domain name to be associated with
            this marker. If not provided the default NVTX domain will be used.
        grad_message: An optional ``string`` message to be associated with
            the op gradient. If not provided ``message`` will be used.
        grad_domain_name: An optional ``string`` domain name to be associated
            with this marker gradient. If not provided ``domain_name`` will
            be used.
        trainable: ``bool``, if ``True`` will make this op
            trainable. Used when this is the first operation in the graph to
            prevent an open ended marker during gradient calculation.
        enabled: ``bool``, if ``False`` the nvtx marker will be disabled.
        name: An optional `string` name for the operation.

    Returns:
        ``tuple``:
        - output: The inputs ``Tensor``.
        - nvtx_context: ``list``, NVTX context associated with this op and passed to :func:`ops.end <end>`. ``None``  if ``enabled=False``.

    """
    if not enabled:
        return inputs, None

    domain_name = domain_name or ''
    grad_message = grad_message or message
    grad_domain_name = grad_domain_name or domain_name or ''

    null_input = 1.
    if trainable:
        with tf.variable_scope("nvtx", reuse=tf.AUTO_REUSE):
            null_input = tf.get_variable('null_input', shape=(),
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer,
                                         trainable=True)

    inputs, marker_id, domain_handle = nvtx_tf_ops.nvtx_start(
        inputs=inputs, null_input=null_input,
        message=message, domain_name=domain_name, name=name)
    return inputs, (marker_id, domain_handle, grad_message, grad_domain_name)


def end(inputs, nvtx_context, name=None):
    """An identity operation with a side effect of closing an NVTX marker.

    Note:
        The :func:`ops.start <start>` and :func:`ops.end <end>` operations
        must be used in pairs.

    Example:
        .. highlight:: python
        .. code-block:: python

            x, nvtx_context = nvtx.plugins.tf.ops.start(x, message='Dense 1-3',
                domain_name='Forward', grad_domain_name='Gradient')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_1')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_2')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_3')
            x = nvtx.plugins.tf.ops.end(x, nvtx_context)

    Arguments:
        inputs: A ``Tensor`` object that will be passed to ``output``.
        nvtx_context: ``list``, NVTX context received from
            :func:`ops.start <start>` If `None` the marker will be disabled.
        name: An optional ``string`` name for the operation.

    Returns:
        The inputs ``Tensor``.

    """
    if nvtx_context is None:
        return inputs

    marker_id, domain_handle, grad_message, grad_domain_name = nvtx_context
    output, null_output = nvtx_tf_ops.nvtx_end(inputs=inputs,
        marker_id=marker_id, domain_handle=domain_handle,
        grad_message=grad_message, grad_domain_name=grad_domain_name, name=name)

    return output


def trace(message, domain_name=None,
          grad_message=None, grad_domain_name=None,
          trainable=False, enabled=True, name=None):
    """An identity function decorator with a side effect of adding NVTX marker.

    Note:
        The decorator expects the wrapped function to take the input ``Tensor``
        as the first argument or to be named ``inputs``, and to return a single
        ``Tensor``.

    Arguments:
        message: A ``string`` message to be associated with this marker.
        domain_name: An optional ``string`` domain name to be associated with
            this marker. If not provided the default NVTX domain will be used.
        grad_message: An optional ``string`` message to be associated with
            the op gradient. If not provided `message` will be used.
        grad_domain_name: An optional ``string`` domain name to be associated
            with this marker gradient. If not provided ``domain_name`` will
            be used.
        trainable: ``bool``, if ``True`` will make this op
            trainable. Used when this is the first operation in the graph to
            prevent an open ended marker during gradient calculation.
        enabled: ``bool``, if ``False`` the nvtx marker will be disabled.
        name: An optional ``string`` name for the operation.

    """
    @wrapt.decorator
    def func_wrapper(wrapped, instance, args, kwargs):
        try:
            inputs = kwargs["inputs"] if "inputs" in kwargs else args[0]
        except:
            raise ValueError("The input tensor must be the first argument"
                             " or named `inputs`")
        assert isinstance(inputs, tf.Tensor)
        start_name = '{}_start'.format(name) if name else None
        end_name = '{}_end'.format(name) if name else None
        inputs, nvtx_context = start(inputs=inputs,
            message=message, domain_name=domain_name,
            grad_message=grad_message, grad_domain_name=grad_domain_name,
            enabled=enabled, trainable=trainable, name=start_name)
        if "inputs" not in kwargs:
            args = [inputs] + list(args[1:])
        else:
            kwargs["inputs"] = inputs
        output = wrapped(*args, **kwargs)
        output = end(inputs=output, nvtx_context=nvtx_context, name=end_name)
        return output

    return func_wrapper
