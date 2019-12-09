#!/usr/bin/env bash

pip install twine

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd ${BASEDIR}/../

bash scripts/clean_repository.sh

python setup.py sdist
python setup.py egg_info

# shellcheck disable=SC2012
twine upload "dist/$(ls -1 dist/ | sort -r | head -n 1)"
