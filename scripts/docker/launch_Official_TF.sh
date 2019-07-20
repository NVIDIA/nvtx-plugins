#!/bin/bash

docker run -it --rm \
  --runtime=nvidia \
  --ipc='host' \
  -v $PWD:/workspace/nvtx \
  tensorflow/tensorflow:latest-gpu-py3 /bin/bash
