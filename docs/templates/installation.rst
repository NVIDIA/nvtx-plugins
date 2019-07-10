Installation
============

NVTX for TF
-----------

Using a prebuilt package
^^^^^^^^^^^^^^^^^^^^^^^^
We have prebuilt packages available for NGC TensorFlow container: https://github.com/NVIDIA/nvtx-plugins/releases

Compiling from source
^^^^^^^^^^^^^^^^^^^^^
Alternatively, you can use the package by compile directly from source with:

.. code-block:: bash

    make pip_pkg

A python wheel will be generated in `artifacts/` which can be installed using
pip:

.. code-block:: bash

    pip install artifacts/nvtx_plugins_tf-*.whl

We recommend building the package inside NVIDIA’s NGC TensorFlow container:
https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow

For more information about how to get started with NVIDIA’s NGC containers,
see the following sections from the NVIDIA GPU Cloud Documentation and the Deep
Learning DGX Documentation: `Getting Started Using NVIDIA GPU
Cloud <https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html>`_,
`Accessing And Pulling From The NGC container registry <https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry>`_
and `Running TensorFlow <https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/index.html>`_.


Build the documentation
^^^^^^^^^^^^^^^^^^^^^^^
The documentation are built by running:

.. code-block:: bash

    pip install -r docs/requirements.txt
    make docs_html

The documentation files will be generated in `docs/build/html`

Building the documentation does not require NVTX Plugins to be installed.
Nonetheless, due to an issue in Sphinx **only Python 3.7 is supported** to build the documentation.



Nsight Systems
--------------

NVIDIA Nsight Systems and can be downloaded and from the
`NVIDIA's Developer Website <https://developer.nvidia.com/nsight-systems>`_. `nsys` is
preinstalled in our NGC TensorFlow container.

More details about nsys and Nsight Systems can be found
`here <https://docs.nvidia.com/nsight-systems/index.html>`_.
