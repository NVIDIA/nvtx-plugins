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

import numpy as np
import tensorflow as tf
import nvtx.plugins.tf as nvtx_tf
from nvtx.plugins.tf.estimator import NVTXHook

ENABLE_NVTX = True
TRAINING_STEPS = 5000


def batch_generator(features, labels, batch_size, steps):
    dataset_len = len(labels)
    idxs = list(range(dataset_len))

    idxs_trunc = None

    steps_per_epoch = dataset_len // batch_size

    for step in range(steps):

        start_idx = batch_size * (step % steps_per_epoch)

        end_idx = batch_size * ((step + 1) % steps_per_epoch)
        end_idx = end_idx if end_idx != 0 else (steps_per_epoch * batch_size)

        if step % (steps_per_epoch) == 0:
            np.random.shuffle(idxs)
            idxs_trunc = idxs[0:batch_size * steps_per_epoch]

        x_batch = np.array([features[j] for j in idxs_trunc[start_idx:end_idx]])

        y_batch = np.array([labels[j] for j in idxs_trunc[start_idx:end_idx]])
        y_batch = np.expand_dims(y_batch, axis=1)

        yield x_batch, y_batch


# Option 1: use decorators
@nvtx_tf.ops.trace(message='Dense Block', domain_name='Forward',
                   grad_domain_name='Gradient', enabled=ENABLE_NVTX, trainable=True)
def DenseBinaryClassificationNet(inputs):
    x = inputs
    x, nvtx_context = nvtx_tf.ops.start(x, message='Dense 1',
        domain_name='Forward', grad_domain_name='Gradient',
        trainable=True, enabled=ENABLE_NVTX)
    x = tf.compat.v1.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_1')
    x = nvtx_tf.ops.end(x, nvtx_context)

    x, nvtx_context = nvtx_tf.ops.start(x, message='Dense 2',
        domain_name='Forward', grad_domain_name='Gradient', enabled=ENABLE_NVTX)
    x = tf.compat.v1.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_2')
    x = nvtx_tf.ops.end(x, nvtx_context)

    x, nvtx_context = nvtx_tf.ops.start(x, message='Dense 3',
        domain_name='Forward', grad_domain_name='Gradient', enabled=ENABLE_NVTX)
    x = tf.compat.v1.layers.dense(x, 512, activation=tf.nn.relu, name='dense_3')
    x = nvtx_tf.ops.end(x, nvtx_context)

    x, nvtx_context = nvtx_tf.ops.start(x, message='Dense 4',
        domain_name='Forward', grad_domain_name='Gradient', enabled=ENABLE_NVTX)
    x = tf.compat.v1.layers.dense(x, 512, activation=tf.nn.relu, name='dense_4')
    x = nvtx_tf.ops.end(x, nvtx_context)

    x, nvtx_context = nvtx_tf.ops.start(x, message='Dense 5',
        domain_name='Forward', grad_domain_name='Gradient', enabled=ENABLE_NVTX)
    x = tf.compat.v1.layers.dense(x, 1, activation=None, name='dense_5')
    x = nvtx_tf.ops.end(x, nvtx_context)

    predictions = x
    return predictions

tf.compat.v1.disable_eager_execution()

# Load Dataset
dataset = np.loadtxt('examples/pima-indians-diabetes.data.csv', delimiter=',')
features = dataset[:, 0:8]
labels = dataset[:, 8]

# tf Graph Inputs
features_plh = tf.compat.v1.placeholder('float', [None, 8])
labels_plh = tf.compat.v1.placeholder('float', [None, 1])


logits = DenseBinaryClassificationNet(inputs=features_plh)
loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_plh))
acc = tf.math.reduce_mean(tf.compat.v1.metrics.accuracy(labels=labels_plh, predictions=tf.round(tf.nn.sigmoid(logits))))
optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(loss)


# Initialize variables. local variables are needed to be initialized for tf.metrics.*
init_g = tf.compat.v1.global_variables_initializer()
init_l = tf.compat.v1.local_variables_initializer()

nvtx_callback = NVTXHook(skip_n_steps=1, name='Train')

# Start training
with tf.compat.v1.train.MonitoredSession(hooks=[nvtx_callback]) as sess:
    sess.run([init_g, init_l])

    # Run graph
    for step, (x, y) in enumerate(batch_generator(features, labels, batch_size=128, steps=TRAINING_STEPS)):
        _, loss_, acc_ = sess.run(
            [optimizer, loss, acc],
            feed_dict={features_plh: x, labels_plh: y}
        )

        if step % 100 == 0:
            print('Step: %04d, loss=%f acc=%f' % (step, loss_, acc_))

    print('\nFinal loss=%f acc=%f' % (loss_, acc_))

print('Optimization Finished!')
