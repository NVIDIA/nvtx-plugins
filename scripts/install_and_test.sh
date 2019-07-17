#!/usr/bin/env bash

set -e  # make sure the script fails and stops at first exception

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${BASEDIR}/../

bash scripts/clean_repository.sh
pip uninstall -y nvtx-plugins


echo -e "\n\n########################################"
echo -e "########################################"
echo -e "          Starting Build Test           "
echo -e "########################################"
echo -e "########################################\n\n"

echo -e "======================================"
echo -e "             Build Command"
echo -e "======================================\n\n"
sleep 1
python setup.py build

echo -e "\n\n======================================"
echo -e "      Testing Build Dist Command"
echo -e "======================================\n\n"
sleep 1
python setup.py bdist

echo -e "\n\n======================================"
echo -e "        Testing Install Command"
echo -e "======================================\n\n"
sleep 1
python setup.py install

echo -e "\n\n======================================"
echo -e "   Testing SDist + Install Command"
echo -e "======================================\n\n"
sleep 1
pip uninstall -y nvtx-plugins
rm -rf dist/ build/
python setup.py sdist
pip install dist/nvtx-plugins*.tar.gz


echo -e "\n\n########################################"
echo -e "########################################"
echo -e "           Starting Run Tests           "
echo -e "########################################"
echo -e "########################################\n\n"
sleep 1

echo -e "======================================"
echo -e "Run Test 1: TF Session"
echo -e "======================================\n\n"
sleep 1
python examples/tf_session_example.py

echo -e "\n\n======================================"
echo -e "Run Test 2: NSYS + TF Session"
echo -e "======================================\n\n"
sleep 1
bash examples/run_tf_session.sh

echo -e "\n\n======================================"
echo -e "Run Test 3: Keras"
echo -e "======================================\n\n"
sleep 1
python examples/keras_example.py

echo -e "\n\n======================================"
echo -e "Run Test 4: NSYS + Keras"
echo -e "======================================\n\n"
sleep 1
bash examples/run_keras.sh


echo -e "\n\n########################################"
echo -e "########################################"
echo -e "             Tests Finished             "
echo -e "########################################"
echo -e "########################################\n\n"