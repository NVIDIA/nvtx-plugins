#!/usr/bin/env bash

export CUDA_LAUNCH_BLOCKING=1
nsys profile -d 60 -w true --sample=cpu -t 'nvtx,cuda' -o examples/keras_example python examples/keras_example.py
