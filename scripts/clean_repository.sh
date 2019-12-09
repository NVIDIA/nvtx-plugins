#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${BASEDIR}/../

rm -rf build/
rm -rf docs/_build/

rm -rf nvtx_plugins/python/*.egg-info/

for i in {1..8}
do
   rm -rf "$(seq -s**/ $i|tr -d '[:digit:]')*.so"
   rm -rf "$(seq -s**/ $i|tr -d '[:digit:]')__pycache__/"
done

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

rm -f examples/*qd*
