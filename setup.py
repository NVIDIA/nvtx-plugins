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
"""Setup for pip package.
"""
import os
import sys

from setuptools import setup

sys.path.append(os.path.join(os.path.dirname(__file__), "nvtx_plugins/python"))

from nvtx.plugins.tf.package_info import __version__
from nvtx.plugins.tf.package_info import __package_name__
from nvtx.plugins.tf.package_info import __contact_names__
from nvtx.plugins.tf.package_info import __contact_emails__
from nvtx.plugins.tf.package_info import __repository_url__
from nvtx.plugins.tf.package_info import __download_url__
from nvtx.plugins.tf.package_info import __description__
from nvtx.plugins.tf.package_info import __license__
from nvtx.plugins.tf.package_info import __keywords__

REQUIRED_PACKAGES = [
    'tensorflow-gpu',
    'wrapt'
]


setup(
    name=__package_name__,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description=__description__,

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

        # Additional Settings
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],

    # Contained modules and scripts.
    install_requires=REQUIRED_PACKAGES,
    packages=['nvtx.plugins.tf', 'nvtx.plugins.tf.keras'],
    package_dir={'': 'nvtx_plugins/python'},

    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,

    # PyPI package information.
    keywords=__keywords__,
)
