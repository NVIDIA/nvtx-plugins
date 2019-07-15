#!/usr/bin/env bash

# GENERATED COMMAND:
# x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro \
# -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time \
# -D_FORTIFY_SOURCE=2 -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart \
# -o build/lib.linux-x86_64-3.5/nvtx/plugins/tf/lib/nvtx_ops.cpython-35m-x86_64-linux-gnu.so \
# -Wl,--version-script=nvtx_plugins.lds -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework

# TARGET COMMAND:
# g++ -I/usr/local/lib/python3.5/dist-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -O2 \
# -std=c++11 -I/usr/local/cuda/include -o nvtx_plugins/python/nvtx/plugins/tf/lib/nvtx_ops.so \
# nvtx_plugins/cc/nvtx_kernels.cc nvtx_plugins/cc/nvtx_ops.cc \
# -shared -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework \
# -L/usr/local/cuda/lib64 -lnvToolsExt

BASE_DIR="/usr/local/lib/python3.5/dist-packages/nvtx_plugins_tf-0.1.0-py3.5-linux-x86_64.egg"

make clean

NVTX_PLUGINS_WITH_TENSORFLOW=1 python setup.py install

cp "${BASE_DIR}/nvtx/plugins/tf/lib/nvtx_ops.cpython-35m-x86_64-linux-gnu.so" \
   "${BASE_DIR}/nvtx/plugins/tf/lib/nvtx_ops.so"

python examples/tf_session_example.py
sleep 2

rm -rf examples/*qd*
bash examples/run_tf_session.sh
sleep 2

rm -rf examples/*qd*
bash examples/run_keras.sh