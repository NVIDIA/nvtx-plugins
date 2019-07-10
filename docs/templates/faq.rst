Frequently Asked Questions
==========================


How is NVTX Plugins different from the built in markers in the NGC TesnorFlow container ?
-----------------------------------------------------------------------------------------

The NVTX markers in the NGC TensorFlow container wrap a single graph node call
and don't modify the graph.
NVTX plugins allows users to add their own markers to highlight
specific layers or parts of their model by adding NVTX nodes to the graph.
The built in TensorFlow markers can be disabled by setting the environment
variable `TF_DISABLE_NVTX_RANGES`.


Can NVTX Plugins be used with eager execution ?
-----------------------------------------------

Yes, the Keras layers fully support eager execution. However, the nvtx markers
are still added and executed at the graph level and not in python.

We plan to add python level calls in the future.


Is there an overhead to using NVTX Plugins ?
--------------------------------------------

NVTX has a small overhead and when no NVTX logger is present this overhead
amounts to an empty function call. However, NVTX Plugins works by adding
nodes to the graph, and therefore has at least the overhead of initializing and
calling an additional TensorFlow operation.
This overhead is small and mostly negligible in large models.

In general, NVTX Plugins is intended for profiling and debugging, it is
not recommended for use in deployed code.


In the example scripts, what does environment variable CUDA_LAUNCH_BLOCKING do ?
--------------------------------------------------------------------------------

The environment variable `CUDA_LAUNCH_BLOCKING` disables asynchronously kernel
launches and is useful for debugging.

More about asynchronous execution at:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution
