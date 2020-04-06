#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${BASEDIR}/../

rm -rf artifacts/
rm -rf build/
rm -rf dist/
rm -rf docs/_build/
rm -rf .pytest_cache/

rm -rf nvtx_plugins/**/*.egg-info/

for i in {1..10}
do
   rm -rf "$(seq -s**/ $i|tr -d '[:digit:]')*.so"
   rm -rf "$(seq -s**/ $i|tr -d '[:digit:]')__pycache__/"
done

find . | grep -E "(__pycache__|\.pyc|\.pyo|\.so$)" | xargs rm -rf

rm -f examples/*.qdrep
rm -f examples/*.qdstm
rm -f examples/*.sqlite
