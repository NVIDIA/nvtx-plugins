CXX := g++
PYTHON_BIN_PATH = python

SRCS = $(wildcard nvtx_plugins/cc/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_HOME ?= /usr/local/cuda
NV_INC ?= $(CUDA_HOME)/include
NV_LIB ?= $(CUDA_HOME)/lib64

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11 -I$(NV_INC)
LDFLAGS = -shared ${TF_LFLAGS} -L$(NV_LIB) -lnvToolsExt

TARGET_LIB = nvtx_plugins/python/nvtx/plugins/tf/lib/nvtx_ops.so
WHEEL_DEST = 'artifacts'


.PHONY: op clean docs
op: $(TARGET_LIB)

$(TARGET_LIB): $(SRCS)
	mkdir -p $(@D)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

pip_pkg: $(TARGET_LIB)
	./build_pip_pkg.sh ${WHEEL_DEST}

docs_html:
	$(MAKE) -C docs html

clean:
	rm -rf $(TARGET_LIB) ${WHEEL_DEST}
	$(MAKE) -C docs clean
