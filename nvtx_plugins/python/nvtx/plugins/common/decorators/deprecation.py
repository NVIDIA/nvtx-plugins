#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect
import sys
import functools
import warnings

import wrapt

from nvtx.plugins import logging

from nvtx.plugins.common.decorators.utils import \
    add_deprecation_notice_to_docstring
from nvtx.plugins.common.decorators.utils import get_qualified_name
from nvtx.plugins.common.decorators.utils import validate_deprecation_args

__all__ = ['deprecated']

# Allow deprecation warnings to be silenced temporarily with a context manager.
_PRINT_DEPRECATION_WARNINGS = True

# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}


def deprecated(wrapped=None, end_support_version=None, instructions='',
               warn_once=True):
    if wrapped is None:
        return functools.partial(
            deprecated,
            end_support_version=end_support_version,
            instructions=instructions,
            warn_once=warn_once
        )

    @wrapt.decorator
    def wrapper(wrapped, instance=None, args=None, kwargs=None):

        validate_deprecation_args(end_support_version, instructions)

        if _PRINT_DEPRECATION_WARNINGS:

            class_or_func_name = get_qualified_name(wrapped)

            if class_or_func_name not in _PRINTED_WARNING:
                if warn_once:
                    _PRINTED_WARNING[class_or_func_name] = True

                if inspect.isclass(wrapped):
                    obj_type = "Class"
                elif inspect.ismethod(wrapped):
                    obj_type = "Method"
                else:
                    obj_type = "Function"

                logging.warning(
                    '{obj_type}: `{module}.{name}` (in file: {file}) is '
                    'deprecated and will be removed in version: `{version}`.\n'
                    'Instructions for updating: {instructions}\n'.format(
                        obj_type=obj_type,
                        module=wrapped.__module__,
                        name=class_or_func_name,
                        file=inspect.getsourcefile(wrapped),
                        version=end_support_version,
                        instructions=instructions
                    )
                )

        return wrapped(*args, **kwargs)

    decorated = wrapper(wrapped)

    if sys.version_info > (3, 0):  # docstring can only be edited with Python 3
        wrapt.FunctionWrapper.__setattr__(
            decorated,
            "__doc__",
            add_deprecation_notice_to_docstring(
                wrapped.__doc__,
                end_support_version,
                instructions
            )
        )

    return decorated


def _deprecated_argument(end_support_version, deprecated_args, is_renaming):
    def rename_kwargs(kwargs, func_name):

        for arg in deprecated_args:

            if arg in kwargs:
                msg = "Deprecated - Inside: `{fn}()`, the argument `{arg}` " \
                      "has been deprecated and will be removed in version " \
                      "`{ver}`.".format(
                    fn=func_name,
                    arg=arg,
                    ver=end_support_version
                )

                if is_renaming:
                    msg += " This argument has been replaced with {}.".format(
                        deprecated_args[arg]
                    )
                else:
                    msg += " Please do not use. See documentation to upgrade."

                warnings.warn(msg, DeprecationWarning)
                logging.warning(msg)

                if is_renaming:
                    kwargs[deprecated_args[arg]] = kwargs.pop(arg)
                else:
                    del kwargs[arg]

    def deco(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):

            try:
                func_name = "{}.{}".format(args[0].__class__.__name__,
                                           f.__name__)
            except (NameError, IndexError):
                func_name = f.__name__

            rename_kwargs(kwargs, func_name)

            return f(*args, **kwargs)

        return wrapper

    return deco


def deprecated_argument(end_support_version, deprecated_args=None):
    if (
            deprecated_args is None or
            not isinstance(deprecated_args, (list, tuple)) or
            len(deprecated_args) == 0 or
            not all([isinstance(x, str) for x in deprecated_args])
    ):
        raise ValueError("`deprecated_aliases` argument should be a list of "
                         "strings, received: %s" % deprecated_args)

    return _deprecated_argument(
        end_support_version,
        deprecated_args,
        is_renaming=False
    )


def deprecated_alias(end_support_version, deprecated_aliases=None):
    if (
            deprecated_aliases is None or
            not isinstance(deprecated_aliases, dict) or
            len(deprecated_aliases) == 0 or
            not all([
                isinstance(x, str) and isinstance(y, str)
                for x, y in deprecated_aliases.items()
            ])
    ):
        raise ValueError("`deprecated_aliases` argument should be a dict of "
                         "string: string, received: %s" % deprecated_aliases)

    return _deprecated_argument(
        end_support_version,
        deprecated_aliases,
        is_renaming=True
    )
