# Build args.
#   * Accepted Values:
#        - Python 3 + Latest GPU: "latest-gpu-py3"  =>  --build-arg TF_CONTAINER_VERSION="latest-gpu-py3"
#        - Python 3 + Latest Devel GPU: "latest-devel-gpu-py3"  =>  --build-arg TF_CONTAINER_VERSION="latest-devel-gpu-py3"
#        - Python 3 + Nightly GPU: "nightly-gpu-py3"  =>  --build-arg TF_CONTAINER_VERSION="nightly-gpu-py3"

# Test Commands:
#
# =============== LATEST STABLE CONTAINER  =============== #
#
# docker build -t nvtx_container --build-arg TF_CONTAINER_VERSION="latest-gpu-py3" - < .circleci/Dockerfile
# docker build -t nvtx_container --build-arg TF_CONTAINER_VERSION="latest-gpu-py3" --build-arg PIP_VERSION="0.1.7" - < .circleci/Dockerfile
# docker build -t nvtx_container --build-arg TF_CONTAINER_VERSION="latest-gpu-py3" --build-arg GIT_REPOSITORY_URL="git@github.com:NVIDIA/nvtx-plugins.git" --build-arg GIT_TAG="0.1.7" - < .circleci/Dockerfile
# docker build -t nvtx_container --build-arg TF_CONTAINER_VERSION="latest-gpu-py3" --build-arg GIT_REPOSITORY_URL="git@github.com:NVIDIA/nvtx-plugins.git" --build-arg GIT_COMMIT_SHA1="f24a12cea34bf1dac398fac7bed7af0e1b239c73" - < .circleci/Dockerfile
#
# =============== NIGHTLY BUILT CONTAINER  =============== #
#
# docker build -t nvtx_container --build-arg TF_CONTAINER_VERSION="nightly-gpu-py3" - < .circleci/Dockerfile
# docker build -t nvtx_container --build-arg TF_CONTAINER_VERSION="nightly-gpu-py3" --build-arg PIP_VERSION="0.1.7" - < .circleci/Dockerfile
# docker build -t nvtx_container --build-arg TF_CONTAINER_VERSION="nightly-gpu-py3" --build-arg GIT_REPOSITORY_URL="git@github.com:NVIDIA/nvtx-plugins.git" --build-arg GIT_TAG="0.1.7" - < .circleci/Dockerfile
# docker build -t nvtx_container --build-arg TF_CONTAINER_VERSION="nightly-gpu-py3" --build-arg GIT_REPOSITORY_URL="git@github.com:NVIDIA/nvtx-plugins.git" --build-arg GIT_COMMIT_SHA1="f24a12cea34bf1dac398fac7bed7af0e1b239c73" - < .circleci/Dockerfile

ARG TF_CONTAINER_VERSION

FROM tensorflow/tensorflow:${TF_CONTAINER_VERSION}

LABEL version="1.0" maintainer="Jonathan DEKHTIAR <contact@jonathandekhtiar.eu>"

# Must Keep to make the Env Var visible
ARG TF_CONTAINER_VERSION

 # On Circle-CI: CIRCLE_SHA1
ARG GIT_COMMIT_SHA1

# On Circle-CI: CIRCLE_TAG
ARG GIT_TAG

# On Circle-CI: CIRCLE_REPOSITORY_URL
ARG GIT_REPOSITORY_URL

ARG PIP_VERSION

RUN printf "\nContainer Tag: ${TF_CONTAINER_VERSION}\n\n" \
    && apt-get update \
    && apt-get install -y git \
    && apt-get install -y g++-$(python -c "import tensorflow as tf; print(tf.version.COMPILER_VERSION[:3])") || true \
    && mkdir -p /dist/ && cd /dist/ \
    && if [[ ! -z "$GIT_REPOSITORY_URL" ]]; then \
        GIT_REPOSITORY_URL="${GIT_REPOSITORY_URL/git@github.com:/https://github.com/}" \
        && printf "\nGit Repository URL: ${GIT_REPOSITORY_URL}\n"; \
    fi \
    && if [[ ! -z "$GIT_COMMIT_SHA1" ]]; then \
        printf "\nBuilding a Commit Release: ${GIT_COMMIT_SHA1}\n\n" \
        && git clone "${GIT_REPOSITORY_URL}" \
        && cd nvtx-plugins && git checkout "${GIT_COMMIT_SHA1}" \
        && pip install \
          --disable-pip-version-check \
          --no-cache-dir \
          --upgrade \
          -e .; \
    elif [[ ! -z "$GIT_TAG" ]]; then \
        printf "\nBuilding a Tag Release: ${GIT_TAG}\n\n" \
        && git clone "${GIT_REPOSITORY_URL}" \
        && cd nvtx-plugins && git checkout "tags/${GIT_TAG}" \
        && pip install \
          --disable-pip-version-check \
          --no-cache-dir \
          --upgrade \
          -e .; \
    elif [[ ! -z "$PIP_VERSION" ]]; then \
        printf "\nBuilding Tag Release: ${PIP_VERSION}\n\n" \
        && git clone https://github.com/NVIDIA/nvtx-plugins.git \
        && cd nvtx-plugins && git checkout "tags/${GIT_TAG}" \
        && pip install \
          --disable-pip-version-check \
          --no-cache-dir \
          --upgrade \
          nvtx-plugins=="${PIP_VERSION}"; \
    else \
        printf "\nBuilding Latest Released Package\n\n" \
        && git clone https://github.com/NVIDIA/nvtx-plugins.git \
        && cd nvtx-plugins && git fetch --tags \
        && git checkout "tags/$(git describe --tags `git rev-list --tags --max-count=1`)" \
        && pip install \
          --disable-pip-version-check \
          --no-cache-dir \
          --upgrade \
          nvtx-plugins; \
    fi \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
