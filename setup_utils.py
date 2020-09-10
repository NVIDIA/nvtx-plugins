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
import subprocess
import sys
import textwrap
import traceback

from contextlib import contextmanager
from copy import deepcopy

from distutils.errors import CCompilerError
from distutils.errors import CompileError
from distutils.errors import DistutilsError
from distutils.errors import DistutilsPlatformError
from distutils.errors import LinkError
from distutils.file_util import copy_file
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion

from setuptools.command.build_ext import build_ext

# Avoid loading Tensorflow or PyTorch during packaging.
if "sdist" not in sys.argv:
    # Torch must be imported first
    try:
        import torch
        from torch.utils.cpp_extension import check_compiler_abi_compatibility
    except ModuleNotFoundError:
        pass

    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        pass

__all__ = [
    "custom_build_ext",
]

# determining if the system has cmake installed
try:
    subprocess.check_output(['cmake', '--version'])
    have_cmake = True

except (FileNotFoundError, subprocess.CalledProcessError):
    have_cmake = False


class VersionError(DistutilsError):
    pass


def parse_version(version_str):

    if "dev" in version_str:
        return 9999999999

    m = re.match('^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?', version_str)

    if m is None:
        return None

    # turn version string to long integer
    version = int(m.group(1)) * 10**9

    if m.group(2) is not None:
        version += int(m.group(2)) * 10**6

    if m.group(3) is not None:
        version += int(m.group(3)) * 10**3

    if m.group(4) is not None:
        version += int(m.group(4))

    return version


def check_tf_version():
    import tensorflow as tf
    if LooseVersion(tf.__version__) < LooseVersion('1.14.0'):
        raise DistutilsPlatformError(
            'Your TensorFlow version %s is outdated.  '
            'NVTX Plugins requires tensorflow>=1.1.0' % tf.__version__
        )


def check_torch_version():
    import torch
    if LooseVersion(torch.__version__) < LooseVersion('1.3.0'):
        raise DistutilsPlatformError(
            'Your PyTorch version %s is outdated.  '
            'Horovod requires torch>=0.4.0' % torch.__version__
        )


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

    extdir = os.path.abspath(os.path.dirname(build_ext.get_ext_fullpath(ext.name)))

    # config = 'Debug' if build_ext.debug else 'Release'
    config = 'Release'

    cmake_args = [
        '-DCMAKE_BUILD_TYPE=' + config,
        '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(config.upper(), extdir),
        '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(config.upper(), lib_output_dir),
    ]

    cmake_build_args = [
        '--config',
        config,
        '--',
        '-j4',
    ]

    # Keep temp build files within a unique subdirectory
    build_temp = os.path.abspath(os.path.join(build_ext.build_temp, ext.name))

    if not os.path.exists(build_temp):
        os.makedirs(build_temp)

    # Config and build the extension
    try:
        subprocess.check_call([cmake_bin, ext.cmake_lists_dir] + cmake_args, cwd=build_temp)
        subprocess.check_call([cmake_bin, '--build', '.'] + cmake_build_args, cwd=build_temp)

    except OSError as e:
        raise RuntimeError('CMake failed: {}'.format(str(e)))

    # Add the library so the plugin will link against it during compilation
    if plugin_ext:
        plugin_ext.libraries += [ext.name]

    if options:
        options['LIBRARIES'] += [ext.name]


def remove_offensive_gcc_compiler_options(compiler_version):
    offensive_replacements = dict()

    if compiler_version < LooseVersion('4.9'):
        offensive_replacements = {'-Wdate-time': '', '-fstack-protector-strong': '-fstack-protector'}

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


# def check_avx_supported():
#     try:
#         flags_output = subprocess.check_output(
#             'gcc -march=native -E -v - </dev/null 2>&1 | grep cc1',
#             shell=True, universal_newlines=True).strip()
#         flags = shlex.split(flags_output)
#         return '+f16c' in flags and '+avx' in flags
#     except subprocess.CalledProcessError:
#         # Fallback to non-AVX if were not able to get flag information.
#         return False


def get_cpp_flags(build_ext):
    last_err = None

    default_flags = ['-std=c++11', '-fPIC', '-O2', '-Wall']

    # avx_flags = ['-mf16c', '-mavx'] if check_avx_supported() else []
    avx_flags = []

    flags_to_try = [default_flags, default_flags + ['-stdlib=libc++']]

    if avx_flags:
        flags_to_try.append(default_flags + avx_flags)
        flags_to_try.append(default_flags + ['-stdlib=libc++'] + avx_flags)

    for cpp_flags in flags_to_try:
        try:
            test_compile(
                build_ext,
                'test_cpp_flags',
                extra_compile_preargs=cpp_flags,
                code=textwrap.dedent(
                    '''\
                    #include <unordered_map>
                    void test() {
                    }
                    '''
                )
            )

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
            test_compile(
                build_ext,
                'test_link_flags',
                extra_link_preargs=link_flags,
                code=textwrap.
                dedent('''\
                    void test() {
                    }
                    ''')
            )

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

    cuda_home = os.environ.get('CUDA_HOME')
    cuda_lib = os.environ.get('CUDA_LIB')
    cuda_include = os.environ.get('CUDA_INCLUDE')

    if cuda_home and os.path.exists(cuda_home):
        for _dir in ['%s/include' % cuda_home]:
            if os.path.exists(_dir):
                cuda_include_dirs.append(_dir)

        for _dir in ['%s/lib' % cuda_home, '%s/lib64' % cuda_home]:
            if os.path.exists(_dir):
                cuda_lib_dirs.append(_dir)

    if cuda_include and os.path.exists(cuda_include) and cuda_include not in cuda_include_dirs:
        cuda_include_dirs.append(cuda_include)

    if cuda_lib and os.path.exists(cuda_lib) and cuda_lib not in cuda_lib_dirs:
        cuda_lib_dirs.append(cuda_lib)

    if not cuda_include_dirs and not cuda_lib_dirs:
        # default to /usr/local/cuda
        cuda_include_dirs += ['/usr/local/cuda/include']
        cuda_lib_dirs += ['/usr/local/cuda/lib', '/usr/local/cuda/lib64']

    try:
        test_compile(
            build_ext,
            'test_cuda',
            libraries=['cudart'],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_lib_dirs,
            extra_compile_preargs=cpp_flags,
            code=textwrap.dedent(
                '''\
                #include <cuda_runtime.h>
                void test() {
                    cudaSetDevice(0);
                }
                '''
            )
        )

    except (CompileError, LinkError):
        raise DistutilsPlatformError(
            'CUDA library was not found (see error above).\n'
            'Please specify correct CUDA location with the CUDA_HOME '
            'environment variable or combination of CUDA_INCLUDE and '
            'CUDA_LIB environment variables.\n\n'
            'CUDA_HOME - path where CUDA include and lib directories can be found\n'
            'CUDA_INCLUDE - path to CUDA include directory\n'
            'CUDA_LIB - path to CUDA lib directory'
        )

    return cuda_include_dirs, cuda_lib_dirs


def get_common_options(build_ext):
    cpp_flags = get_cpp_flags(build_ext)
    link_flags = get_link_flags(build_ext)

    have_cuda = True

    MACROS = []

    INCLUDES = []

    SOURCES = []

    COMPILE_FLAGS = cpp_flags
    LINK_FLAGS = link_flags

    LIBRARY_DIRS = []
    LIBRARIES = []

    if have_cuda:
        cuda_include_dirs, cuda_lib_dirs = get_cuda_dirs(build_ext, cpp_flags)
        MACROS += [('HAVE_CUDA', '1')]
        INCLUDES += cuda_include_dirs
        LIBRARY_DIRS += cuda_lib_dirs
        LIBRARIES += ['cudart']

    return dict(
        MACROS=MACROS,
        INCLUDES=INCLUDES,
        SOURCES=SOURCES,
        COMPILE_FLAGS=COMPILE_FLAGS,
        LINK_FLAGS=LINK_FLAGS,
        LIBRARY_DIRS=LIBRARY_DIRS,
        LIBRARIES=LIBRARIES
    )


# run the customize_compiler
class custom_build_ext(build_ext):

    @contextmanager
    def _filter_build_errors(self, ext):

        def warn(msg):
            msg = "WARNING: %s" % msg
            print("\n%s" % ("%" * (len(msg))))
            print(msg)
            print("%s\n" % ("%" * (len(msg))))

        try:
            yield

        # except (CCompilerError, DistutilsError, CompileError) as e:
        #     if not ext.optional:
        #         raise
        #
        #     warn('building extension "%s" failed: %s' % (ext.name, e))

        except ModuleNotFoundError as e:
            if not ext.optional:
                raise
            warn('extension "%s" has been skipped\nReason: %s' % (ext.name, e))

        except:
            raise CompileError('Unable to build plugin, will skip it.\n\n%s' % traceback.format_exc())

    def copy_extensions_to_source(self):
        build_py = self.get_finalized_command('build_py')

        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)

            modpath = fullname.split('.')
            package = '.'.join(modpath[:-1])

            package_dir = build_py.get_package_dir(package)

            dest_filename = os.path.join(package_dir, os.path.basename(filename))
            src_filename = os.path.join(self.build_lib, filename)

            if ext._built_with_success:
                # Always copy, even if source is older than destination, to ensure
                # that the right extensions for the current Python/platform are
                # used.
                copy_file(src_filename, dest_filename, verbose=self.verbose, dry_run=self.dry_run)

                if ext._needs_stub:
                    self.write_stub(package_dir or os.curdir, ext, True)

    def build_extensions(self):

        options = get_common_options(self)

        for extension in self.extensions:

            with self._filter_build_errors(extension):

                if extension.__class__.__name__ == "TFExtension":
                    build_tf_extension(self, extension, options)
                    extension._built_with_success = True

                elif extension.__class__.__name__ == "PyTExtension":
                    build_torch_extension(self, extension, options)
                    extension._built_with_success = True

                else:
                    raise CompileError("Unsupported extension: %s", extension)


def build_tf_extension(build_ext, ext, options):
    check_tf_version()

    # Backup the options, preventing other plugins access libs that
    # compiled with compiler of this plugin
    options = deepcopy(options)

    tf_compile_flags, tf_link_flags = get_tf_flags(build_ext, options['COMPILE_FLAGS'])

    # Update HAVE_CUDA to mean that PyTorch supports CUDA.
    ext.define_macros = options['MACROS'] + ext.define_macros
    ext.include_dirs = options['INCLUDES'] + [
        x() if callable(x) else x for x in ext.include_dirs
    ]

    ext.sources = options['SOURCES'] + ext.sources

    ext.extra_compile_args = options['COMPILE_FLAGS'] + tf_compile_flags + ext.extra_compile_args
    ext.extra_link_args = options['LINK_FLAGS'] + tf_link_flags + ext.extra_link_args

    ext.library_dirs = options['LIBRARY_DIRS'] + ext.library_dirs
    ext.libraries = options['LIBRARIES'] + ext.libraries

    cc_compiler = cxx_compiler = None

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

                candidate_cc_compiler = find_matching_gcc_compiler_in_path(candidate_compiler_version)

                if candidate_cc_compiler and candidate_compiler_version > compiler_version:
                    cc_compiler = candidate_cc_compiler
                    cxx_compiler = candidate_cxx_compiler
                    compiler_version = candidate_compiler_version

            else:
                print(
                    "==================================================================================================="
                )
                print(
                    'INFO: Compiler %s (version %s) is not usable for this TensorFlow '
                    'installation. Require g++ (version >=%s, <%s).' %
                    (candidate_cxx_compiler, candidate_compiler_version, tf_compiler_version, maximum_compiler_version)
                )
                print(
                    "==================================================================================================="
                )

        if cc_compiler:
            print("===================================================================================================")
            print(
                'INFO: Compilers %s and %s (version %s) selected for TensorFlow plugin build.' %
                (cc_compiler, cxx_compiler, compiler_version)
            )
            print("===================================================================================================")

        else:
            raise DistutilsPlatformError(
                'Could not find compiler compatible with this TensorFlow installation.\n'
                'Please check the Github repository for recommended compiler versions.\n'
                'To force a specific compiler version, set CC and CXX environment variables.'
            )

        cflags, cppflags, ldshared = remove_offensive_gcc_compiler_options(compiler_version)

        try:
            with env(CC=cc_compiler, CXX=cxx_compiler, CFLAGS=cflags, CPPFLAGS=cppflags, LDSHARED=ldshared):
                customize_compiler(build_ext.compiler)

                try:
                    build_ext.compiler.compiler.remove("-DNDEBUG")
                except (AttributeError, ValueError):
                    pass

                try:
                    build_ext.compiler.compiler_so.remove("-DNDEBUG")
                except (AttributeError, ValueError):
                    pass

                try:
                    build_ext.compiler.compiler_so.remove("-Wstrict-prototypes")
                except (AttributeError, ValueError):
                    pass

                try:
                    build_ext.compiler.linker_so.remove("-Wl,-O1")
                except (AttributeError, ValueError):
                    pass

                # import pprint
                # print('########################################################')
                # print('########################################################')
                # pprint.pprint(dir(build_ext))
                # print('########################################################')
                # print('########################################################')
                # pprint.pprint(build_ext.__dict__)
                # print('########################################################')
                # print('########################################################')

                build_ext.build_extension(ext)
        finally:
            # Revert to the default compiler settings
            customize_compiler(build_ext.compiler)


# def build_torch_extension_v2(build_ext, global_options, torch_version):
def build_torch_extension(build_ext, ext, options):
    check_torch_version()

    import copy

    def _add_compile_flag(extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_torch_extension_name(extension):
        define = '-DTORCH_EXTENSION_NAME={}'.format(extension.name.split(".")[-1])
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(define)
        else:
            extension.extra_compile_args.append(define)

    def _add_gnu_cpp_abi_flag(extension):
        # use the same CXX ABI as what PyTorch was compiled with
        _add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))

    _add_compile_flag(ext, '-DTORCH_API_INCLUDE_EXTENSION_H')
    _define_torch_extension_name(ext)
    _add_gnu_cpp_abi_flag(ext)

    try:
        import cffi
        if LooseVersion(cffi.__version__) < LooseVersion('1.4.0'):
            raise VersionError("torch.utils.ffi requires cffi version >= 1.4, but got %s" % cffi.__version__)
    except ImportError:
        raise ImportError("torch.utils.ffi requires the cffi package")

    # Backup the options, preventing other plugins access libs that
    # compiled with compiler of this plugin
    options = deepcopy(options)

    # Versions of PyTorch > 1.3.0 require C++14
    pyt_compile_flags = set_flag(options['COMPILE_FLAGS'], 'std', 'c++14')
    check_torch_is_built_with_cuda(build_ext, include_dirs=options['INCLUDES'], extra_compile_args=pyt_compile_flags)

    pyt_updated_macros = set_macros(
        macros=options['MACROS'],
        keys_and_values=[
            # Update HAVE_CUDA to mean that PyTorch supports CUDA.
            ('HAVE_CUDA', "1"),
            # Export TORCH_VERSION equal to our representation of torch.__version__.
            # Internally it's used for backwards compatibility checks.
            ('TORCH_VERSION', str(parse_version(torch.__version__))),
            # Always set _GLIBCXX_USE_CXX11_ABI, since PyTorch can only detect whether it was set to 1.
            ('_GLIBCXX_USE_CXX11_ABI', str(int(torch.compiled_with_cxx11_abi()))),
            # PyTorch requires -DTORCH_API_INCLUDE_EXTENSION_H
            ('TORCH_API_INCLUDE_EXTENSION_H', '1')
        ]
    )

    ext.define_macros = pyt_updated_macros + ext.define_macros
    ext.include_dirs = options['INCLUDES'] + ext.include_dirs

    ext.sources = options['SOURCES'] + ext.sources

    ext.extra_compile_args = options['COMPILE_FLAGS'] + pyt_compile_flags + ext.extra_compile_args
    ext.extra_link_args = options['LINK_FLAGS'] + pyt_compile_flags + ext.extra_link_args

    ext.library_dirs = options['LIBRARY_DIRS'] + ext.library_dirs
    ext.libraries = options['LIBRARIES'] + ext.libraries

    cc_compiler = cxx_compiler = cflags = cppflags = ldshared = None

    if sys.platform.startswith('linux') and not os.getenv('CC') and not os.getenv('CXX'):

        # Find the compatible compiler of the highest version
        compiler_version = LooseVersion('0')

        for candidate_cxx_compiler, candidate_compiler_version in find_gxx_compiler_in_path():

            if check_compiler_abi_compatibility(candidate_cxx_compiler):
                candidate_cc_compiler = find_matching_gcc_compiler_in_path(candidate_compiler_version)

                if candidate_cc_compiler and candidate_compiler_version > compiler_version:
                    cc_compiler = candidate_cc_compiler
                    cxx_compiler = candidate_cxx_compiler
                    compiler_version = candidate_compiler_version

            else:
                print(
                    "==================================================================================================="
                )
                print(
                    'INFO: Compiler %s (version %s) is not usable for this PyTorch '
                    'installation, see the warning above.' % (candidate_cxx_compiler, candidate_compiler_version)
                )
                print(
                    "==================================================================================================="
                )

        if cc_compiler:
            print("===================================================================================================")
            print(
                'INFO: Compilers %s and %s (version %s) selected for PyTorch plugin build.' %
                (cc_compiler, cxx_compiler, compiler_version)
            )
            print("===================================================================================================")

        else:
            raise DistutilsPlatformError(
                'Could not find compiler compatible with this PyTorch installation.\n'
                'Please check the Github repository for recommended compiler versions.\n'
                'To force a specific compiler version, set CC and CXX environment variables.'
            )

        cflags, cppflags, ldshared = remove_offensive_gcc_compiler_options(compiler_version)

    try:
        with env(CC=cc_compiler, CXX=cxx_compiler, CFLAGS=cflags, CPPFLAGS=cppflags, LDSHARED=ldshared):
            customize_compiler(build_ext.compiler)

            try:
                build_ext.compiler.compiler.remove("-DNDEBUG")
            except (AttributeError, ValueError):
                pass

            try:
                build_ext.compiler.compiler_so.remove("-DNDEBUG")
            except (AttributeError, ValueError):
                pass

            try:
                build_ext.compiler.compiler_so.remove("-Wstrict-prototypes")
            except (AttributeError, ValueError):
                pass

            try:
                build_ext.compiler.linker_so.remove("-Wl,-O1")
            except (AttributeError, ValueError):
                pass

            build_ext.build_extension(ext)
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


def check_macro(macros, key):
    return any(k == key and v for k, v in macros)


def set_macros(macros, keys_and_values):

    for key, new_value in keys_and_values:
        macros = set_macro(macros, key, new_value)

    return macros


def set_macro(macros, key, new_value):
    if any(k == key for k, _ in macros):
        return [(k, new_value if k == key else v) for k, v in macros]
    else:
        return macros + [(key, new_value)]


def set_flag(flags, flag, value):
    flag = '-' + flag
    if any(f.split('=')[0] == flag for f in flags):
        return [('{}={}'.format(flag, value) if f.split('=')[0] == flag else f) for f in flags]
    else:
        return flags + ['{}={}'.format(flag, value)]


def get_tf_include_dirs():
    import tensorflow as tf
    tf_inc = tf.sysconfig.get_include()
    return [tf_inc, '%s/external/nsync/public' % tf_inc]


def get_tf_lib_dirs():
    import tensorflow as tf
    tf_lib = tf.sysconfig.get_lib()
    return [tf_lib]


def test_compile(
    build_ext,
    name,
    code,
    libraries=None,
    include_dirs=None,
    library_dirs=None,
    macros=None,
    extra_compile_preargs=None,
    extra_link_preargs=None
):
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

    compiler.compile([source_file], extra_preargs=extra_compile_preargs, include_dirs=include_dirs, macros=macros)

    compiler.link_shared_object(
        [object_file],
        shared_object_file,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_preargs=extra_link_preargs
    )

    return shared_object_file


def get_tf_abi(build_ext, include_dirs, lib_dirs, libs, cpp_flags):
    cxx11_abi_macro = '_GLIBCXX_USE_CXX11_ABI'

    for cxx11_abi in ['0', '1']:
        try:
            lib_file = test_compile(
                build_ext,
                'test_tensorflow_abi',
                macros=[(cxx11_abi_macro, cxx11_abi)],
                include_dirs=include_dirs,
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_preargs=cpp_flags,
                code=textwrap.dedent(
                    '''\
                #include <string>
                #include "tensorflow/core/framework/op.h"
                #include "tensorflow/core/framework/op_kernel.h"
                #include "tensorflow/core/framework/shape_inference.h"
                void test() {
                    auto ignore = tensorflow::strings::StrCat("a", "b");
                }
                '''
                )
            )

            from tensorflow.python.framework import load_library
            load_library.load_op_library(lib_file)

            return cxx11_abi_macro, cxx11_abi
        except (CompileError, LinkError):
            last_err = 'Unable to determine CXX11 ABI to use with TensorFlow (see error above).'
        except Exception:
            last_err = 'Unable to determine CXX11 ABI to use with TensorFlow.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


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
            ''')
            )

            from tensorflow.python.framework import load_library
            load_library.load_op_library(lib_file)

            return tf_libs

        except (CompileError, LinkError):
            last_err = 'Unable to determine -l link flags to use with TensorFlow (see error above).'

        except Exception:
            last_err = 'Unable to determine -l link flags to use with TensorFlow.  Last error:\n\n%s' % \
                       traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def check_torch_is_built_with_cuda(build_ext, include_dirs, extra_compile_args):
    from torch.utils.cpp_extension import include_paths
    test_compile(
        build_ext,
        'test_torch_cuda',
        include_dirs=include_dirs + include_paths(cuda=True),
        extra_compile_preargs=extra_compile_args,
        code=textwrap.dedent('''\
        #include <THC/THC.h>
        void test() {
        }
        ''')
    )


def get_tf_flags(build_ext, cpp_flags):
    import tensorflow as tf

    try:
        return tf.sysconfig.get_compile_flags(), tf.sysconfig.get_link_flags()

    except AttributeError:
        # fallback to the previous logic
        tf_include_dirs = get_tf_include_dirs()
        tf_lib_dirs = get_tf_lib_dirs()
        tf_libs = get_tf_libs(build_ext, tf_lib_dirs, cpp_flags)
        tf_abi = get_tf_abi(build_ext, tf_include_dirs, tf_lib_dirs, tf_libs, cpp_flags)

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
            '%s -dM -E - </dev/null' % compiler, shell=True, universal_newlines=True
        ).split('\n')

        for m in compiler_macros:
            version_match = re.match('^#define __VERSION__ "(.*?)"$', m)
            if version_match:
                return LooseVersion(version_match.group(1))

        print("===========================================================================================")
        print('INFO: Unable to determine version of the gcc compiler %s.' % compiler)
        print("===========================================================================================")

    except subprocess.CalledProcessError:
        print("===========================================================================================")
        print('INFO: Unable to determine version of the gcc compiler %s.\n%s' % (compiler, traceback.format_exc()))
        print("===========================================================================================")

    return None


def determine_nvcc_version(compiler):
    try:
        nvcc_macros = [
            _l for _l in
            subprocess.check_output('%s --version </dev/null' %
                                    compiler, shell=True, universal_newlines=True).split('\n') if _l != ''
        ][-1]

        nvcc_version = nvcc_macros.split(", ")[-1][1:]
        return LooseVersion(nvcc_version)

    except subprocess.CalledProcessError:
        print("===========================================================================================")
        print('INFO: Unable to determine version of the nvcc compiler %s.\n%s' % (compiler, traceback.format_exc()))
        print("===========================================================================================")

    return None


def enumerate_binaries_in_path():
    for path_dir in os.getenv('PATH', '').split(':'):
        if os.path.isdir(path_dir):
            for bin_file in sorted(os.listdir(path_dir)):
                yield path_dir, bin_file


def find_matching_gcc_compiler_in_path(gxx_compiler_version):
    for path_dir, bin_file in enumerate_binaries_in_path():
        if re.match('^gcc(?:-\\d+(?:\\.\\d+)*)?$', bin_file):
            # gcc, or gcc-7, gcc-4.9, or gcc-4.8.5
            compiler = os.path.join(path_dir, bin_file)
            compiler_version = determine_gcc_version(compiler)

            if compiler_version == gxx_compiler_version:
                return compiler

    else:
        print("===========================================================================================")
        print('INFO: Unable to find gcc compiler (version %s).' % gxx_compiler_version)
        print("===========================================================================================")


def find_gxx_compiler_in_path():
    compilers = []

    for path_dir, bin_file in enumerate_binaries_in_path():
        if re.match('^g\\+\\+(?:-\\d+(?:\\.\\d+)*)?$', bin_file):
            # g++, or g++-7, g++-4.9, or g++-4.8.5
            compiler = os.path.join(path_dir, bin_file)
            compiler_version = determine_gcc_version(compiler)

            if compiler_version:
                compilers.append((compiler, compiler_version))

    if not compilers:
        print("===========================================================================================")
        print('INFO: Unable to find any gxx compiler.')
        print("===========================================================================================")

    return compilers


def find_nvcc_compiler_in_path():

    for path_dir, bin_file in enumerate_binaries_in_path():

        if bin_file == 'nvcc':
            compiler = os.path.join(path_dir, bin_file)
            compiler_version = determine_nvcc_version(compiler)

            return compiler, compiler_version

    else:
        print("===========================================================================================")
        print('INFO: Unable to find any nvcc compiler.')
        print("===========================================================================================")
