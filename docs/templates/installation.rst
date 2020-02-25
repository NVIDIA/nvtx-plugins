Installation
============

Installing NVTX-Plugins
^^^^^^^^^^^^^^^^^^^^^^^
The package can be installed from PyPI:

.. code-block:: bash

    # Stable release
    pip install nvtx-plugins

    # Pre-release (may present bugs)
    pip install nvtx-plugins --pre

The package is also available for download on github: https://github.com/NVIDIA/nvtx-plugins/releases

.. code-block:: bash

    pip install nvtx-plugins*.tar.gz

Installing from github
^^^^^^^^^^^^^^^^^^^^^^

You can build and install the package from the github repository:

.. code-block:: bash
    # Install Master Branch
    pip install git+https://github.com/NVIDIA/nvtx-plugins

    # Install Specific Commit (In this case commit 7d46c3a)
    pip install git+https://github.com/NVIDIA/nvtx-plugins@7d46c3a

    # Install Specific Branch (In this case branch master)
    pip install git+https://github.com/NVIDIA/nvtx-plugins@master

    # Install Specific Release (In this case 0.1.7)
    pip install git+https://github.com/NVIDIA/nvtx-plugins@0.1.7

Installing from source
^^^^^^^^^^^^^^^^^^^^^^

You can build and install the package from source:

.. code-block:: bash

    python setup.py sdist
    pip install dist/nvtx-plugins*.tar.gz

For development objectives, you can install the package directly from source with:

.. code-block:: bash

    python setup.py install

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
The documentation is built by running:

.. code-block:: bash

    cd docs
    pip install -r requirements.txt
    make html

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
