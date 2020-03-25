#!/usr/bin/env bash

TF_CPP_MIN_LOG_LEVEL="3"

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${BASEDIR}/../

# Clean Prebuilt Packages
rm -rf dist/
rm -rf rm -rf nvtx_plugins/python/nvtx_plugins.egg-info/
python setup.py sdist

rm -rf **/__pycache__
rm -rf **/*.pyc

pip install -r requirements/requirements_test.txt
pip install --no-cache-dir --upgrade dist/nvtx-plugins-*.tar.gz

rm -f examples/*.qdrep
rm -f examples/*.sqlite

# pytest --full-trace
python -m tests.test_keras
python -m tests.test_tensorflow_session
