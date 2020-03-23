# Build args.
#   * Accepted Values:
#        - Python 3 + Latest GPU: "latest-gpu-py3"  =>  --build-arg TF_CONTAINER_VERSION="latest-gpu-py3"
#        - Python 3 + Latest Devel GPU: "latest-devel-gpu-py3"  =>  --build-arg TF_CONTAINER_VERSION="latest-devel-gpu-py3"
#        - Python 3 + Nightly GPU: "nightly-gpu-py3"  =>  --build-arg TF_CONTAINER_VERSION="nightly-gpu-py3"

ARG TF_CONTAINER_VERSION

FROM tensorflow/tensorflow:${TF_CONTAINER_VERSION}

LABEL version="1.0" maintainer="Jonathan DEKHTIAR <contact@jonathandekhtiar.eu>"

ARG PIP_VERSION
ARG TF_CONTAINER_VERSION

RUN echo "Container Tag: ${TF_CONTAINER_VERSION}" \
    && apt-get update \
    && if [ -z "$PIP_VERSION" ]; then \
        echo "Building a Nightly Release" \
        && apt-get install -y git \
        && mkdir /dist/ && cd /dist/ \
        && git clone https://github.com/NVIDIA/nvtx-plugins.git \
        && cd nvtx-plugins \
        && pip install --disable-pip-version-check --no-cache-dir --upgrade -e .; \
    else \
        echo "Building Tag Release: $PIP_VERSION" \
        && pip install  --disable-pip-version-check --no-cache-dir --upgrade nvtx-plugins=="${PIP_VERSION }"; \
    fi \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
