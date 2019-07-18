# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Code Inspired by: https://github.com/horovod/horovod/blob/master/setup.py
#
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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
#
# Modified by NVIDIA to fit our requirements
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import os
import re
import shlex
import subprocess
import sys
import textwrap
import traceback

from contextlib import contextmanager

from distutils.errors import CompileError
from distutils.errors import DistutilsError
from distutils.errors import DistutilsPlatformError
from distutils.errors import LinkError
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion

from setuptools.command.build_ext import build_ext


__all__ = ["custom_build_ext"]


# determining if the system has cmake installed
have_cmake = True
try:
    subprocess.check_output(['cmake', '--version'])
except:
    have_cmake = False


def check_tf_version():
    try:
        import tensorflow as tf
        if LooseVersion(tf.__version__) < LooseVersion('1.1.0'):
            raise DistutilsPlatformError(
                'Your TensorFlow version %s is outdated.  '
                'NVTX Plugins requires tensorflow>=1.1.0' % tf.__version__)
    except ImportError:
        raise DistutilsPlatformError(
            'import tensorflow failed, is it installed?\n\n%s' % traceback.format_exc())
    except AttributeError:
        # This means that tf.__version__ was not exposed, which makes it *REALLY* old.
        raise DistutilsPlatformError(
            'Your TensorFlow version is outdated.  NVTX Plugins requires tensorflow>=1.1.0')


def build_cmake(build_ext, ext, prefix, plugin_ext=None, options=None):

    cmake_bin = 'cmake'

    # All statically linked libraries will be placed here
    lib_output_dir = os.path.abspath(os.path.join(build_ext.build_temp, 'lib', prefix))

    if not os.path.exists(lib_output_dir):
        os.makedirs(lib_output_dir)

    if plugin_ext:
        plugin_ext.library_dirs += [lib_output_dir]

    if options:
        options['LIBRARY_DIRS'] += [lib_output_dir]

    extdir = os.path.abspath(
        os.path.dirname(build_ext.get_ext_fullpath(ext.name)))
    config = 'Debug' if build_ext.debug else 'Release'
    cmake_args = [
        '-DUSE_MPI=ON',
        '-DCMAKE_BUILD_TYPE=' + config,
        '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(config.upper(), extdir),
        '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(config.upper(),
                                                        lib_output_dir),
    ]
    cmake_build_args = [
        '--config', config,
        '--', '-j4',
    ]

    # Keep temp build files within a unique subdirectory
    build_temp = os.path.abspath(os.path.join(build_ext.build_temp, ext.name))
    if not os.path.exists(build_temp):
        os.makedirs(build_temp)

    # Config and build the extension
    try:
        subprocess.check_call([cmake_bin, ext.cmake_lists_dir] + cmake_args,
                              cwd=build_temp)
        subprocess.check_call([cmake_bin, '--build', '.'] + cmake_build_args,
                              cwd=build_temp)
    except OSError as e:
        raise RuntimeError('CMake failed: {}'.format(str(e)))

    # Add the library so the plugin will link against it during compilation
    if plugin_ext:
        plugin_ext.libraries += [ext.name]

    if options:
        options['LIBRARIES'] += [ext.name]


def find_matching_gcc_compiler_path(gxx_compiler_version):

    for path_dir, bin_file in enumerate_binaries_in_path():
        if re.match('^gcc(?:-\\d+(?:\\.\\d+)*)?$', bin_file):
            # gcc, or gcc-7, gcc-4.9, or gcc-4.8.5
            compiler = os.path.join(path_dir, bin_file)
            compiler_version = determine_gcc_version(compiler)

            if compiler_version == gxx_compiler_version:
                return compiler

    print("=========================================================================")
    print('INFO: Unable to find gcc compiler (version %s).' % gxx_compiler_version)
    print("===========================================================================================")
    return None


def remove_offensive_gcc_compiler_options(compiler_version):
    offensive_replacements = dict()
    if compiler_version < LooseVersion('4.9'):
        offensive_replacements = {
            '-Wdate-time': '',
            '-fstack-protector-strong': '-fstack-protector'
        }

    if offensive_replacements:
        from sysconfig import get_config_var
        cflags = get_config_var('CONFIGURE_CFLAGS')
        cppflags = get_config_var('CONFIGURE_CPPFLAGS')
        ldshared = get_config_var('LDSHARED')

        for k, v in offensive_replacements.items():
            cflags = cflags.replace(k, v)
            cppflags = cppflags.replace(k, v)
            ldshared = ldshared.replace(k, v)

        return cflags, cppflags, ldshared

    # Use defaults
    return None, None, None


def check_avx_supported():
    try:
        flags_output = subprocess.check_output(
            'gcc -march=native -E -v - </dev/null 2>&1 | grep cc1',
            shell=True, universal_newlines=True).strip()
        flags = shlex.split(flags_output)
        return '+f16c' in flags and '+avx' in flags
    except subprocess.CalledProcessError:
        # Fallback to non-AVX if were not able to get flag information.
        return False


def get_cpp_flags(build_ext):

    last_err = None

    default_flags = ['-std=c++11', '-fPIC', '-O2', '-Wall']

    avx_flags = ['-mf16c', '-mavx'] if check_avx_supported() else []

    flags_to_try = [default_flags + avx_flags,
                    default_flags + ['-stdlib=libc++'] + avx_flags,
                    default_flags,
                    default_flags + ['-stdlib=libc++']]

    for cpp_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_cpp_flags',
                         extra_compile_preargs=cpp_flags,
                         code=textwrap.dedent('''\
                    #include <unordered_map>
                    void test() {
                    }
                    '''))

            return cpp_flags

        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ compilation flags (see error above).'

        except Exception:
            last_err = 'Unable to determine C++ compilation flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_link_flags(build_ext):

    last_err = None

    libtool_flags = ['-Wl,-exported_symbols_list']
    ld_flags = []

    flags_to_try = [ld_flags, libtool_flags]

    for link_flags in flags_to_try:

        try:
            test_compile(build_ext, 'test_link_flags',
                         extra_link_preargs=link_flags,
                         code=textwrap.dedent('''\
                    void test() {
                    }
                    '''))

            return link_flags

        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ link flags (see error above).'

        except Exception:
            last_err = 'Unable to determine C++ link flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_cuda_dirs(build_ext, cpp_flags):

    cuda_include_dirs = []
    cuda_lib_dirs = []

    cuda_home = os.environ.get('NVTX_PLUGINS_CUDA_HOME')
    if cuda_home:
        cuda_include_dirs += ['%s/include' % cuda_home]
        cuda_lib_dirs += ['%s/lib' % cuda_home, '%s/lib64' % cuda_home]

    cuda_include = os.environ.get('NVTX_PLUGINS_CUDA_INCLUDE')
    if cuda_include:
        cuda_include_dirs += [cuda_include]

    cuda_lib = os.environ.get('NVTX_PLUGINS_CUDA_LIB')
    if cuda_lib:
        cuda_lib_dirs += [cuda_lib]

    if not cuda_include_dirs and not cuda_lib_dirs:
        # default to /usr/local/cuda
        cuda_include_dirs += ['/usr/local/cuda/include']
        cuda_lib_dirs += ['/usr/local/cuda/lib', '/usr/local/cuda/lib64']

    try:
        test_compile(build_ext, 'test_cuda', libraries=['cudart'],
                     include_dirs=cuda_include_dirs,
                     library_dirs=cuda_lib_dirs,
                     extra_compile_preargs=cpp_flags,
                     code=textwrap.dedent('''\
            #include <cuda_runtime.h>
            void test() {
                cudaSetDevice(0);
            }
            '''))

    except (CompileError, LinkError):
        raise DistutilsPlatformError(
            'CUDA library was not found (see error above).\n'
            'Please specify correct CUDA location with the NVTX_PLUGINS_CUDA_HOME '
            'environment variable or combination of NVTX_PLUGINS_CUDA_INCLUDE and '
            'NVTX_PLUGINS_CUDA_LIB environment variables.\n\n'
            'NVTX_PLUGINS_CUDA_HOME - path where CUDA include and lib directories can be found\n'
            'NVTX_PLUGINS_CUDA_INCLUDE - path to CUDA include directory\n'
            'NVTX_PLUGINS_CUDA_LIB - path to CUDA lib directory')

    return cuda_include_dirs, cuda_lib_dirs


def get_common_options(build_ext):

    cpp_flags = get_cpp_flags(build_ext)
    link_flags = get_link_flags(build_ext)

    have_cuda = True
    cuda_include_dirs, cuda_lib_dirs = get_cuda_dirs(build_ext, cpp_flags)

    MACROS = []

    INCLUDES = []

    SOURCES = []

    COMPILE_FLAGS = cpp_flags
    LINK_FLAGS = link_flags

    LIBRARY_DIRS = []
    LIBRARIES = []

    if have_cuda:
        MACROS += [('HAVE_CUDA', '1')]
        INCLUDES += cuda_include_dirs
        LIBRARY_DIRS += cuda_lib_dirs
        LIBRARIES += ['cudart']

    return dict(MACROS=MACROS,
                INCLUDES=INCLUDES,
                SOURCES=SOURCES,
                COMPILE_FLAGS=COMPILE_FLAGS,
                LINK_FLAGS=LINK_FLAGS,
                LIBRARY_DIRS=LIBRARY_DIRS,
                LIBRARIES=LIBRARIES)


# run the customize_compiler
class custom_build_ext(build_ext):

    def __init__(self, dist, tf_lib):

        self._tf_lib = tf_lib
        super(custom_build_ext, self).__init__(dist)

    def build_extensions(self):

        options = get_common_options(self)

        built_plugins = []

        try:
            if not os.environ.get('NVTX_PLUGINS_WITHOUT_TENSORFLOW', False):
                build_tf_extension(self, self._tf_lib, options)
                built_plugins.append(True)

            else:
                print("===========================================================================================")
                print(
                    'INFO: TensorFlow plugin building is skipped, remove the environment variable: `%s`.\n\n' %
                    'NVTX_PLUGINS_WITHOUT_TENSORFLOW',
                    file=sys.stderr
                )
                print("===========================================================================================")

                built_plugins.append(False)

        except:
            print("===========================================================================================")
            print(
                'INFO: Unable to build TensorFlow plugin, will skip it.\n\n%s' % traceback.format_exc(),
                file=sys.stderr
            )
            print("===========================================================================================")

            built_plugins.append(False)

        if not built_plugins:
            raise DistutilsError('TensorFlow NVTX plugin was excluded from build. Aborting.')

        if not any(built_plugins):
            raise DistutilsError('No plugin was built. See errors above.')


def build_tf_extension(build_ext, tf_lib, options):
    check_tf_version()

    tf_compile_flags, tf_link_flags = get_tf_flags(build_ext, options['COMPILE_FLAGS'])

    tf_lib.define_macros = options['MACROS']
    tf_lib.include_dirs = options['INCLUDES']

    tf_lib.sources = options['SOURCES'] + tf_lib.sources

    tf_lib.extra_compile_args = options['COMPILE_FLAGS'] + tf_compile_flags
    tf_lib.extra_link_args = options['LINK_FLAGS'] + tf_link_flags

    tf_lib.library_dirs = options['LIBRARY_DIRS']
    tf_lib.libraries = options['LIBRARIES']

    cc_compiler = cxx_compiler = cflags = cppflags = None

    if not sys.platform.startswith('linux'):
        raise EnvironmentError("Only Linux Systems are supported")

    if not os.getenv('CC') and not os.getenv('CXX'):
        # Determine g++ version compatible with this TensorFlow installation
        import tensorflow as tf

        if hasattr(tf, 'version'):
            # Since TensorFlow 1.13.0
            tf_compiler_version = LooseVersion(tf.version.COMPILER_VERSION)

        else:
            tf_compiler_version = LooseVersion(tf.COMPILER_VERSION)

        if tf_compiler_version.version[0] == 4:
            # g++ 4.x is ABI-incompatible with g++ 5.x+ due to std::function change
            # See: https://github.com/tensorflow/tensorflow/issues/27067
            maximum_compiler_version = LooseVersion('5')

        else:
            maximum_compiler_version = LooseVersion('999')

        # Find the compatible compiler of the highest version
        compiler_version = LooseVersion('0')

        for candidate_cxx_compiler, candidate_compiler_version in find_gxx_compiler_in_path():

            if tf_compiler_version <= candidate_compiler_version < maximum_compiler_version:

                candidate_cc_compiler = find_matching_gcc_compiler_path(candidate_compiler_version)

                if candidate_cc_compiler and candidate_compiler_version > compiler_version:
                    cc_compiler = candidate_cc_compiler
                    cxx_compiler = candidate_cxx_compiler
                    compiler_version = candidate_compiler_version

            else:
                print("===========================================================================================")
                print(
                    'INFO: Compiler %s (version %s) is not usable for this TensorFlow '
                    'installation. Require g++ (version >=%s, <%s).' %
                    (candidate_cxx_compiler, candidate_compiler_version, tf_compiler_version, maximum_compiler_version)
                )
                print("===========================================================================================")

        if cc_compiler:
            print("===========================================================================================")
            print('INFO: Compilers %s and %s (version %s) selected for TensorFlow plugin build.' % (
                cc_compiler, cxx_compiler, compiler_version
            ))
            print("===========================================================================================")

        else:
            raise DistutilsPlatformError(
                'Could not find compiler compatible with this TensorFlow installation.\n'
                'Please check the NVTX-Plugins Github Repository for recommended compiler versions.\n'
                'To force a specific compiler version, set CC and CXX environment variables.')

        cflags, cppflags, ldshared = remove_offensive_gcc_compiler_options(compiler_version)

        try:
            with env(CC=cc_compiler, CXX=cxx_compiler, CFLAGS=cflags, CPPFLAGS=cppflags, LDSHARED=ldshared):
                customize_compiler(build_ext.compiler)

                try:
                    build_ext.compiler.compiler_so.remove("-Wstrict-prototypes")
                except (AttributeError, ValueError):
                    pass

                build_ext.build_extension(tf_lib)
        finally:
            # Revert to the default compiler settings
            customize_compiler(build_ext.compiler)


@contextmanager
def env(**kwargs):
    # ignore args with None values
    for k in list(kwargs.keys()):
        if kwargs[k] is None:
            del kwargs[k]

    # backup environment
    backup = {}
    for k in kwargs.keys():
        backup[k] = os.environ.get(k)

    # set new values & yield
    for k, v in kwargs.items():
        os.environ[k] = v

    try:
        yield
    finally:
        # restore environment
        for k in kwargs.keys():
            if backup[k] is not None:
                os.environ[k] = backup[k]
            else:
                del os.environ[k]


def get_tf_include_dirs():
    import tensorflow as tf
    tf_inc = tf.sysconfig.get_include()
    return [tf_inc, '%s/external/nsync/public' % tf_inc]


def get_tf_lib_dirs():
    import tensorflow as tf
    tf_lib = tf.sysconfig.get_lib()
    return [tf_lib]


def test_compile(build_ext, name, code, libraries=None, include_dirs=None, library_dirs=None, macros=None,
                 extra_compile_preargs=None, extra_link_preargs=None):

    test_compile_dir = os.path.join(build_ext.build_temp, 'test_compile')

    if not os.path.exists(test_compile_dir):
        os.makedirs(test_compile_dir)

    source_file = os.path.join(test_compile_dir, '%s.cc' % name)
    with open(source_file, 'w') as f:
        f.write(code)

    compiler = build_ext.compiler

    [object_file] = compiler.object_filenames([source_file])
    shared_object_file = compiler.shared_object_filename(name, output_dir=test_compile_dir)

    try:
        build_ext.compiler.compiler_so.remove("-Wstrict-prototypes")
    except (AttributeError, ValueError):
        pass

    compiler.compile(
        [source_file],
        extra_preargs=extra_compile_preargs,
        include_dirs=include_dirs,
        macros=macros
    )

    compiler.link_shared_object(
        [object_file],
        shared_object_file,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_preargs=extra_link_preargs
    )

    return shared_object_file


def get_tf_libs(build_ext, lib_dirs, cpp_flags):

    for tf_libs in [['tensorflow_framework'], []]:
        try:
            lib_file = test_compile(
                build_ext,
                'test_tensorflow_libs',
                library_dirs=lib_dirs,
                libraries=tf_libs,
                extra_compile_preargs=cpp_flags,
                code=textwrap.dedent('''\
                void test() {
                }
            '''))

            from tensorflow.python.framework import load_library
            load_library.load_op_library(lib_file)

            return tf_libs

        except (CompileError, LinkError):
            last_err = 'Unable to determine -l link flags to use with TensorFlow (see error above).'

        except Exception:
            last_err = 'Unable to determine -l link flags to use with TensorFlow.  Last error:\n\n%s' % \
                       traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_tf_abi(build_ext, include_dirs, lib_dirs, libs, cpp_flags):

    cxx11_abi_macro = '_GLIBCXX_USE_CXX11_ABI'

    for cxx11_abi in ['0', '1']:
        try:
            lib_file = test_compile(build_ext, 'test_tensorflow_abi',
                                    macros=[(cxx11_abi_macro, cxx11_abi)],
                                    include_dirs=include_dirs,
                                    library_dirs=lib_dirs,
                                    libraries=libs,
                                    extra_compile_preargs=cpp_flags,
                                    code=textwrap.dedent('''\
                #include <string>
                #include "tensorflow/core/framework/op.h"
                #include "tensorflow/core/framework/op_kernel.h"
                #include "tensorflow/core/framework/shape_inference.h"
                void test() {
                    auto ignore = tensorflow::strings::StrCat("a", "b");
                }
                '''))

            from tensorflow.python.framework import load_library
            load_library.load_op_library(lib_file)

            return cxx11_abi_macro, cxx11_abi
        except (CompileError, LinkError):
            last_err = 'Unable to determine CXX11 ABI to use with TensorFlow (see error above).'
        except Exception:
            last_err = 'Unable to determine CXX11 ABI to use with TensorFlow.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_tf_flags(build_ext, cpp_flags):

    import tensorflow as tf

    try:
        return tf.sysconfig.get_compile_flags(), tf.sysconfig.get_link_flags()

    except AttributeError:
        # fallback to the previous logic
        tf_include_dirs = get_tf_include_dirs()
        tf_lib_dirs = get_tf_lib_dirs()
        tf_libs = get_tf_libs(build_ext, tf_lib_dirs, cpp_flags)
        tf_abi = get_tf_abi(build_ext, tf_include_dirs,
                            tf_lib_dirs, tf_libs, cpp_flags)

        compile_flags = []
        for include_dir in tf_include_dirs:
            compile_flags.append('-I%s' % include_dir)
        if tf_abi:
            compile_flags.append('-D%s=%s' % tf_abi)

        link_flags = []
        for lib_dir in tf_lib_dirs:
            link_flags.append('-L%s' % lib_dir)
        for lib in tf_libs:
            link_flags.append('-l%s' % lib)

        return compile_flags, link_flags


def determine_gcc_version(compiler):
    try:
        compiler_macros = subprocess.check_output(
            '%s -dM -E - </dev/null' % compiler,
            shell=True,
            universal_newlines=True
        ).split('\n')

        for m in compiler_macros:
            version_match = re.match('^#define __VERSION__ "(.*?)"$', m)
            if version_match:
                return LooseVersion(version_match.group(1))

        print("===========================================================================================")
        print('INFO: Unable to determine version of the compiler %s.' % compiler)
        print("===========================================================================================")

    except subprocess.CalledProcessError:
        print("===========================================================================================")
        print('INFO: Unable to determine version of the compiler %s.\n%s' % (compiler, traceback.format_exc()))
        print("===========================================================================================")

    return None


def enumerate_binaries_in_path():
    for path_dir in os.getenv('PATH', '').split(':'):
        if os.path.isdir(path_dir):
            for bin_file in sorted(os.listdir(path_dir)):
                yield path_dir, bin_file


def find_gxx_compiler_in_path():
    compilers = []

    for path_dir, bin_file in enumerate_binaries_in_path():
        if re.match('^g\\+\\+(?:-\\d+(?:\\.\\d+)*)?$', bin_file):
            # g++, or g++-7, g++-4.9, or g++-4.8.5
            compiler = os.path.join(path_dir, bin_file)
            compiler_version = determine_gcc_version(compiler)
            if compiler_version:
                compilers.append((compiler, compiler_version))

    return compilers
