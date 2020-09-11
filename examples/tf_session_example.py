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

import os
import pathlib

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import nvtx.plugins.tf as nvtx_tf
from nvtx.plugins.tf.estimator import NVTXHook

ENABLE_NVTX = True
TRAINING_STEPS = 1500

tf.compat.v1.disable_eager_execution()


def get_dataset(batch_size):
  # load pima indians dataset
  csv_data = np.loadtxt(
      os.path.join(
          pathlib.Path(__file__).parent.absolute(),
          'pima-indians-diabetes.data.csv'
      ),
      delimiter=','
  )

  features_arr = csv_data[:, 0:8]
  labels_arr = csv_data[:, 8]
  labels_arr = np.expand_dims(labels_arr, axis=1)

  ds_x = tf.data.Dataset.from_tensor_slices(features_arr)
  ds_y = tf.data.Dataset.from_tensor_slices(labels_arr)

  ds = tf.data.Dataset.zip((ds_x, ds_y))

  ds = ds.shuffle(buffer_size=batch_size*2).repeat()
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds


# Option 1: use decorators
@nvtx_tf.ops.trace(message='Dense Block',
                   grad_message='Dense Block grad',
                   domain_name='Trace_Forward',
                   grad_domain_name='Trace_Gradient',
                   enabled=ENABLE_NVTX,
                   trainable=True)
def DenseBinaryClassificationNet(inputs):
    x = inputs
    x, nvtx_context = nvtx_tf.ops.start(x, message='Dense 1',
        grad_message='Dense 1 grad', domain_name='Forward',
        grad_domain_name='Gradient', trainable=True, enabled=ENABLE_NVTX)
    x = tf.compat.v1.layers.dense(x, 1024, activation=tf.nn.relu, name='dense1')
    x = nvtx_tf.ops.end(x, nvtx_context)

    x, nvtx_context = nvtx_tf.ops.start(
        x,
        message='Dense 2',
        grad_message='Dense 2 grad',
        domain_name='Forward',
        grad_domain_name='Gradient',
        enabled=ENABLE_NVTX
    )
    x = tf.compat.v1.layers.dense(x, 1024, activation=tf.nn.relu, name='dense2')
    x = nvtx_tf.ops.end(x, nvtx_context)

    x, nvtx_context = nvtx_tf.ops.start(
        x,
        message='Dense 3',
        grad_message='Dense 3 grad',
        domain_name='Forward',
        grad_domain_name='Gradient',
        enabled=ENABLE_NVTX
    )
    x = tf.compat.v1.layers.dense(x, 512, activation=tf.nn.relu, name='dense3')
    x = nvtx_tf.ops.end(x, nvtx_context)

    x, nvtx_context = nvtx_tf.ops.start(
        x,
        message='Dense 4',
        grad_message='Dense 4 grad',
        domain_name='Forward',
        grad_domain_name='Gradient',
        enabled=ENABLE_NVTX
    )
    x = tf.compat.v1.layers.dense(x, 512, activation=tf.nn.relu, name='dense4')
    x = nvtx_tf.ops.end(x, nvtx_context)

    x, nvtx_context = nvtx_tf.ops.start(
        x,
        message='Dense 5',
        grad_message='Dense 5 grad',
        domain_name='Forward',
        grad_domain_name='Gradient',
        enabled=ENABLE_NVTX
    )
    x = tf.compat.v1.layers.dense(x, 1, activation=None, name='dense5')
    x = nvtx_tf.ops.end(x, nvtx_context)

    predictions = x
    return predictions


if __name__ == "__main__":

    # Load Dataset
    dataset = get_dataset(128)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    features, labels = iterator.get_next()

    logits = DenseBinaryClassificationNet(inputs=features)
    loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=labels
    ))
    acc = tf.math.reduce_mean(tf.compat.v1.metrics.accuracy(
        labels=labels,
        predictions=tf.round(tf.nn.sigmoid(logits))
    ))
    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=0.01,
        momentum=0.9,
        use_nesterov=True
    )

    train_op = optimizer.minimize(loss)

    # Initialize variables. local variables are needed to be initialized
    # for tf.metrics.*
    init_g = tf.compat.v1.global_variables_initializer()
    init_l = tf.compat.v1.local_variables_initializer()

    nvtx_callback = NVTXHook(skip_n_steps=1, name='Train')

    # Start training
    with tf.compat.v1.train.MonitoredSession(hooks=[nvtx_callback]) as sess:
        sess.run([init_g, init_l])

        # Run graph
        for step in range(TRAINING_STEPS):
            _, loss_, acc_ = sess.run([train_op, loss, acc])

            if step % 100 == 0:
                print('Step: %04d, loss=%f acc=%f' % (step, loss_, acc_))

        print('\nFinal loss=%f acc=%f' % (loss_, acc_))

    print('Optimization Finished!')
