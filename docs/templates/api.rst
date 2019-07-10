API
===


TensorFlow ops
--------------

.. autofunction:: nvtx.plugins.tf.ops.start

.. autofunction:: nvtx.plugins.tf.ops.end

.. autodecorator:: nvtx.plugins.tf.ops.trace


Session hooks
-------------

.. autoclass:: nvtx.plugins.tf.estimator.NVTXHook


Keras Layers
------------

.. autoclass:: nvtx.plugins.tf.keras.layers.NVTXStart

.. autoclass:: nvtx.plugins.tf.keras.layers.NVTXEnd


Keras Callbacks
---------------

.. autoclass:: nvtx.plugins.tf.keras.callbacks.NVTXCallback
