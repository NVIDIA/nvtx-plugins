#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${BASEDIR}/../

rm -rf build/ dist/ docs/_build/
rm -rf nvtx_plugins/python/*.egg-info/

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

rm -f examples/*qd*
