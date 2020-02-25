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
NUM_EPOCHS = 200

def batch_generator(features, labels, batch_size=128):
    dataset_len = len(labels)
    idxs = list(range(dataset_len))
    np.random.shuffle(idxs)
    for i in range(dataset_len // batch_size):
        features_batch = [features[j] for j in idxs[batch_size*i:batch_size*(i+1)]]
        label_batch = [labels[j] for j in idxs[batch_size*i:batch_size*(i+1)]]
        label_batch = np.expand_dims(label_batch, axis=1)
        yield (features_batch, features_batch), label_batch


# Option 1: use decorators
@nvtx_tf.ops.trace(message='Dense Block', domain_name='Forward',
                   grad_domain_name='Gradient', enabled=ENABLE_NVTX, trainable=True)
def DenseBinaryClassificationNet(inputs):

    x1, x2 = inputs

    (x1, x2), nvtx_context = nvtx_tf.ops.start(inputs=(x1, x2), message='Dense 1',
        domain_name='Forward', grad_domain_name='Gradient',
        trainable=True, enabled=ENABLE_NVTX)

    net1 = tf.compat.v1.layers.dense(x1, 1024, activation=tf.nn.relu, name='dense_1_1')
    net2 = tf.compat.v1.layers.dense(x2, 1024, activation=tf.nn.relu, name='dense_1_2')

    net1, net2 = nvtx_tf.ops.end([net1, net2], nvtx_context)

    x = tf.concat([net1, net2], axis=-1)

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
    probs = tf.sigmoid(x)

    return predictions, probs

tf.compat.v1.disable_eager_execution()

# Load Dataset
dataset = np.loadtxt('examples/pima-indians-diabetes.data.csv', delimiter=',')
features = dataset[:,0:8]
labels = dataset[:,8]

# tf Graph Inputs
features_plh_1 = tf.compat.v1.placeholder('float', [None, 8])
features_plh_2 = tf.compat.v1.placeholder('float', [None, 8])
labels_plh = tf.compat.v1.placeholder('float', [None, 1])


logits, probs = DenseBinaryClassificationNet(inputs=(features_plh_1, features_plh_2))

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
    for epoch in range(NUM_EPOCHS):
        for (x1, x2), y in batch_generator(features, labels, batch_size=128):
            optimizer_, loss_, acc_ = sess.run(
                [optimizer, loss, acc],
                feed_dict={
                    features_plh_1: x1,
                    features_plh_2: x2,
                    labels_plh: y
                }
            )
        print('Epoch: %d. loss=%f acc=%f' % (epoch+1, loss_, acc_))

    print('Optimization Finished!')
