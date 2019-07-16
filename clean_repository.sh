#!/usr/bin/env bash

rm -rf build/ dist/ docs/_build/
rm -rf nvtx_plugins/python/*.egg-info/

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

rm -f examples/*qd*
