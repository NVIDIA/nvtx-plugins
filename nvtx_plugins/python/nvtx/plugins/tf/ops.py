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

from nvtx.plugins.common.decorators import deprecated_alias
from nvtx.plugins.tf import nvtx_tf_ops

__all__ = ['start', 'end', 'trace']


def _maybe_convert_list_to_tensor(inputs):

    inputs_were_processed = False

    if isinstance(inputs, (list, tuple)) and \
        all([isinstance(x, tf.Tensor) for x in inputs]):
        inputs = tf.stack(inputs, axis=0, name="nvtx_trace_inputs")
        inputs_were_processed = True

    assert isinstance(inputs, tf.Tensor)

    return inputs, inputs_were_processed


@ops.RegisterGradient('NvtxStart')
def _nvtx_start_grad(op, grad, marker_id):
    # grad_message and grad_category_name are not used
    if not isinstance(marker_id, tf.Tensor) and marker_id is None:
        raise RuntimeError('Error in nvtx range %s. '
                           'Make sure all nvtx ranges are closed' % op.name)

    grad, null_grad = nvtx_tf_ops.nvtx_end(
        inputs=grad,
        marker_id=marker_id,
        grad_message=op.inputs[2],
        grad_category_name=op.inputs[3]
    )
    return [grad, null_grad, None, None]


@ops.RegisterGradient('NvtxEnd')
def _nvtx_end_grad(op, grad, null_grad):
    grad, marker_id = nvtx_tf_ops.nvtx_start(
        inputs=grad,
        null_input=1.,
        message=op.inputs[2],
        category_name=op.inputs[3]
    )
    return [grad, marker_id, None, None]


@deprecated_alias(
    deprecated_aliases={
        "domain_name": "category",
        "grad_message": "backward_message",
        "grad_domain_name": "backward_category"
    }
)
def start(inputs, message, category="forward",
          backward_message=None, backward_category="backward",
          trainable=False, enabled=True, name=None):
    """An identity operation with a side effect of opening an NVTX marker.

    Note:
        The :func:`ops.start <start>` and :func:`ops.end <end>` operations
        must be used in pairs.

    Example:
        .. highlight:: python
        .. code-block:: python

            x, nvtx_context = nvtx.plugins.tf.ops.start(x, message='Dense 1-3',
                category_name='Forward', grad_category_name='Gradient')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_1')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_2')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_3')
            x = nvtx.plugins.tf.ops.end(x, nvtx_context)

    Arguments:
        inputs: A ``Tensor`` object that is passed to ``output``.
        message: A ``string`` message to be associated with this marker.
        category: An optional ``string`` NVTX category name to be
            associated with this marker. If not provided the default NVTX
            `forward` category will be used.
        backward_message: An optional ``string`` message to be associated with
            the op gradient. If not provided ``message`` will be used.
        backward_category: An optional ``string`` NVTX category name to be
            associated with this marker gradient. If not provided the default
            NVTX `backward` category will be used.
        trainable: ``bool``, if ``True`` will make this op
            trainable. Used when this is the first operation in the graph to
            prevent an open ended marker during gradient calculation.
        enabled: ``bool``, if ``False`` the nvtx marker will be disabled.
        name: An optional `string` name for the operation.

    Returns:
        ``tuple``:
        - output: The inputs ``Tensor``.
        - nvtx_context: ``list``, NVTX context associated with this op and
        passed to :func:`ops.end <end>`. ``None``  if ``enabled=False``.

    """
    if not enabled:
        return inputs, None

    backward_message = backward_message or message

    null_input = 1.

    if trainable:
        with tf.compat.v1.variable_scope("nvtx", reuse=tf.compat.v1.AUTO_REUSE):
            null_input = tf.compat.v1.get_variable(
                'null_input',
                shape=(),
                dtype=tf.float32,
                initializer=tf.zeros_initializer,
                trainable=True
            )

    inputs, should_unstack = _maybe_convert_list_to_tensor(inputs)

    inputs, marker_id = nvtx_tf_ops.nvtx_start(
        inputs=inputs,
        null_input=null_input,
        message=message,
        category_name=category,
        name=name
    )

    if should_unstack:
        inputs = tf.unstack(inputs, axis=0)

    return inputs, (marker_id, backward_message, backward_category)


def end(inputs, nvtx_context, name=None):
    """An identity operation with a side effect of closing an NVTX marker.

    Note:
        The :func:`ops.start <start>` and :func:`ops.end <end>` operations
        must be used in pairs.

    Example:
        .. highlight:: python
        .. code-block:: python

            x, nvtx_context = nvtx.plugins.tf.ops.start(x, message='Dense 1-3',
                category_name='Forward', grad_category_name='Gradient')
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

    marker_id, backward_message, backward_category = nvtx_context

    inputs, should_unstack = _maybe_convert_list_to_tensor(inputs)

    output, null_output = nvtx_tf_ops.nvtx_end(
        inputs=inputs,
        marker_id=marker_id,
        grad_message=backward_message,
        grad_category_name=backward_category,
        name=name
    )

    if should_unstack:
        output = tf.unstack(output, axis=0)

    return output


@deprecated_alias(
    deprecated_aliases={
        "domain_name": "category",
        "grad_message": "backward_message",
        "grad_domain_name": "backward_category"
    }
)
def trace(message, category=None,
          backward_message=None, backward_category=None,
          trainable=False, enabled=True, name=None):
    """An identity function decorator with a side effect of adding NVTX marker.

    Note:
        The decorator expects the wrapped function to take the input ``Tensor``
        as the first argument or to be named ``inputs``, and to return a single
        ``Tensor``.

    Arguments:
        message: A ``string`` message to be associated with this marker.
        category: An optional ``string`` NVTX category name to be
            associated with this marker. If not provided the default NVTX
            `forward` category will be used.
        backward_message: An optional ``string`` message to be associated with
            the op gradient. If not provided `message` will be used.
        backward_category: An optional ``string`` NVTX category name to be
            associated with this marker gradient. If not provided the default
            NVTX `backward` category will be used.
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

        inputs, should_unstack = _maybe_convert_list_to_tensor(inputs)

        start_name = '{}_start'.format(name) if name else None
        end_name = '{}_end'.format(name) if name else None

        inputs, nvtx_context = start(
            inputs=inputs,
            message=message,
            category=category,
            backward_message=backward_message,
            backward_category=backward_category,
            enabled=enabled,
            trainable=trainable,
            name=start_name
        )

        if should_unstack:
            inputs = tf.unstack(inputs, axis=0)

        if "inputs" in kwargs:
            kwargs["inputs"] = inputs
        else:
            args = [inputs] + list(args[1:])

        output = wrapped(*args, **kwargs)
        output = end(inputs=output, nvtx_context=nvtx_context, name=end_name)

        return output

    return func_wrapper
