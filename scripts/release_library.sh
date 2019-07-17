#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${BASEDIR}/../

bash scripts/clean_repository.sh

python setup.py sdist
python setup.py egg_info

twine upload dist/*
