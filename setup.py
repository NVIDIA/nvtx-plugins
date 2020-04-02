# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup for pip package."""

import codecs
import os
import shutil
import sys
import sysconfig

import subprocess

from setuptools import setup
from setuptools import Command

from nvtx.package_info import __contact_emails__
from nvtx.package_info import __contact_names__
from nvtx.package_info import __description__
from nvtx.package_info import __download_url__
from nvtx.package_info import __homepage__
from nvtx.package_info import __keywords__
from nvtx.package_info import __license__
from nvtx.package_info import __package_name__
from nvtx.package_info import __repository_url__
from nvtx.package_info import __version__

# Note: Do not remove these two lines, they define all the C Extensions
from nvtx.c_extensions_utils import CustomExtension
from nvtx.c_extensions import *

from setup_utils import custom_build_ext

# Force Build Cache Cleaning
def force_clean_build_cache():
    def distutils_dir_name(dname):

        """Returns the name of a distutils build directory"""
        f = "{dirname}.{platform}-{version[0]}.{version[1]}"
        return f.format(dirname=dname,
                        platform=sysconfig.get_platform(),
                        version=sys.version_info)

    for build_dir in ['lib', 'temp']:
        buildpath = os.path.join('build', distutils_dir_name(build_dir), 'nvtx')
        print("BUILD PATH:", buildpath)
        try:
            shutil.rmtree(buildpath)
        except FileNotFoundError:
            pass

force_clean_build_cache()

ext_modules = [x for x in CustomExtension.get_instances()]

class VersionCommand(Command):

    description = 'print library version'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(__version__)


def run_cmd(command):
    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return ps.communicate()[0].decode('utf-8').strip()


def get_tf_pkgname():

    for pkg_name in ["tensorflow-gpu", "tf-nightly-gpu"]:
        if pkg_name in run_cmd("pip freeze | grep %s" % pkg_name):
            return pkg_name

    return "tensorflow"  # Default if not found


def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename)) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt") + [get_tf_pkgname()]
extras_require = {
    'test': req_file("requirements_test.txt"),
}

tests_requirements = extras_require["test"]



# =================== Reading Readme file as TXT files ===================

if os.path.exists('README.rst'):
    # codec is used for consistent encoding
    long_description = codecs.open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'),
        'r', 'utf-8'
    ).read()

    long_description = long_description.replace(
        "docs/images/",
        "https://github.com/NVIDIA/nvtx-plugins/raw/master/docs/images/"
    )

else:
    long_description = 'See ' + __homepage__

setup(
    name=__package_name__,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description=__description__,
    long_description=long_description,

    # The project's main homepage.
    url=__repository_url__,
    download_url=__download_url__,

    # Author details
    author=__contact_names__,
    author_email=__contact_emails__,

    # maintainer Details
    maintainer=__contact_names__,
    maintainer_email=__contact_emails__,

    # The licence under which the project is released
    license=__license__,
    classifiers=[
        # How mature is this project? Common values are
        #  1 - Planning
        #  2 - Pre-Alpha
        #  3 - Alpha
        #  4 - Beta
        #  5 - Production/Stable
        #  6 - Mature
        #  7 - Inactive
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',

        # Indicate what your project relates to
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Supported python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',

        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
    ],

    cmdclass={
        'build_ext': custom_build_ext,
        'version': VersionCommand,
    },
    ext_modules=[tensorflow_nvtx_lib],

    # Add in any packaged data.
    include_package_data=True,
    packages=['nvtx.plugins.tf', 'nvtx.plugins.tf.keras'],

    # Contained modules and scripts.
    install_requires=install_requires,
    extras_require=extras_require,
    setup_requires=['pytest-runner'],

    zip_safe=False,

    # PyPI package information.
    keywords=__keywords__,
)
