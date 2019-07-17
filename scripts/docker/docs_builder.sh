#!/bin/bash

docker run -it --rm \
  -v $PWD:/workspace/nvtx \
  python:3.7 /bin/bash
