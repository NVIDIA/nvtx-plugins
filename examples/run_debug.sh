#!/usr/bin/env bash

export CUDA_LAUNCH_BLOCKING=1

nsys profile \
  -d 60 \
  -w true \
  --force-overwrite=true \
  --sample=cpu \
  -t 'nvtx,cuda' \
  --stop-on-exit=true \
  --kill=sigkill \
  -o examples/debug_nvtx \
  python examples/debug_nvtx.py
