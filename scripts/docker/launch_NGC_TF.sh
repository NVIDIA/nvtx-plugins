#!/bin/bash

docker run -it --rm \
  --runtime=nvidia \
  --ipc='host' \
  -v "$(pwd):/workspace/nvtx" \
  --workdir /workspace/nvtx/ \
  nvcr.io/nvidia/tensorflow:19.12-tf1-py3 /bin/bash
