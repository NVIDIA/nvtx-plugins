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

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from nvtx.plugins.tf.keras.layers import NVTXStart, NVTXEnd
from nvtx.plugins.tf.keras.callbacks import NVTXCallback

NUM_EPOCHS = 200

# load pima indians dataset
dataset = np.loadtxt('examples/pima-indians-diabetes.data.csv', delimiter=',')
features = dataset[:,0:8]
labels = dataset[:,8]


def DenseBinaryClassificationNet(input_shape=(8,)):
    inputs = Input(input_shape)

    x = inputs
    x, marker_id, domain_id = NVTXStart(message='Dense 1',
                                        domain_name='forward',
                                        trainable=True)(x)
    x = Dense(1024, activation='relu')(x)
    x = NVTXEnd(grad_message='Dense 1 grad',
                grad_domain_name='backwards')([x, marker_id, domain_id])


    x, marker_id, domain_id = NVTXStart(message='Dense 2',
                                        domain_name='forward')(x)
    x = Dense(1024, activation='relu')(x)
    x = NVTXEnd(grad_message='Dense 2 grad',
                grad_domain_name='backwards')([x, marker_id, domain_id])


    x, marker_id, domain_id = NVTXStart(message='Dense 3',
                                        domain_name='forward')(x)
    x = Dense(512, activation='relu')(x)
    x = NVTXEnd(grad_message='Dense 3 grad',
                grad_domain_name='backwards')([x, marker_id, domain_id])


    x, marker_id, domain_id = NVTXStart(message='Dense 4',
                                        domain_name='forward')(x)
    x = Dense(512, activation='relu')(x)
    x = NVTXEnd(grad_message='Dense 4 grad',
                grad_domain_name='backwards')([x, marker_id, domain_id])

    x, marker_id, domain_id = NVTXStart(message='Dense 5',
                                        domain_name='forward')(x)
    x = Dense(1, activation='sigmoid')(x)
    x = NVTXEnd(grad_message='Dense 5 grad',
                grad_domain_name='backwards')([x, marker_id, domain_id])

    predictions = x
    model = Model(inputs=inputs, outputs=predictions)
    return model


nvtx_callback = NVTXCallback()

model = DenseBinaryClassificationNet()
sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(features, labels, batch_size=128, epochs=NUM_EPOCHS,
          callbacks=[nvtx_callback])
