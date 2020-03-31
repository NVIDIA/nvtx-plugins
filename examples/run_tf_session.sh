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
  -o examples/tf_session_example \
  python examples/tf_session_example.py
