#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
NOTE: DO NOT REMOVE THESE FILES. They are copied from TensorFlow repository and
are necessary to build the library without installing TensorFlow.

Source: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/util

They replace the following imports:
>>> from tensorflow.python.util import decorator_utils
>>> from tensorflow.python.util.deprecation import _validate_deprecation_args
"""

import sys
import re

__all__ = [
  "add_deprecation_notice_to_docstring",
  "get_qualified_name",
  "validate_deprecation_args"
]


def add_deprecation_notice_to_docstring(doc, version, instructions):
    return _add_deprecated_function_notice_to_docstring(
        doc,
        version,
        instructions
    )


def get_qualified_name(function):
    # Python 3
    if hasattr(function, '__qualname__'):
        return function.__qualname__

    # Python 2
    if hasattr(function, 'im_class'):
        return function.im_class.__name__ + '.' + function.__name__

    return function.__name__


def validate_deprecation_args(version, instructions):
    if version is not None and not re.match(r'^\d+(\.\d+){1,3}$', version):
        raise ValueError('Version must be in format: 1.0, 1.0.0, or 1.0.0.0.')
    if not instructions:
        raise ValueError(
            'Don\'t deprecate things without conversion instructions!'
        )


def _add_deprecated_function_notice_to_docstring(doc, version, instructions):
    """Adds a deprecation notice to a docstring for deprecated functions."""

    if instructions:
        deprecation_message = """
            .. warning::
                **THIS FUNCTION IS DEPRECATED:** It will be removed in {ver}.
                *Instructions for updating:* {instructions}.
        """.format(
            ver=(
                "version: `%s`" % version
                if version is not None else
                "a future version"
            ),
            instructions=instructions
        )

    else:
        deprecation_message = """
            .. warning::
                **THIS FUNCTION IS DEPRECATED:** It will be removed in {ver}.
        """.format(
            ver=(
                "version: `%s`" % version
                if version is not None else
                "a future version"
            )
        )

    main_text = [deprecation_message]

    return _add_notice_to_docstring(doc, 'DEPRECATED FUNCTION', main_text)


def _add_notice_to_docstring(doc, no_doc_str, notice):
    """Adds a deprecation notice to a docstring."""
    if not doc:
        lines = [no_doc_str]

    else:
        lines = _normalize_docstring(doc).splitlines()

    notice = [''] + notice

    if len(lines) > 1:
        # Make sure that we keep our distance from the main body
        if lines[1].strip():
            notice.append('')

        lines[1:1] = notice
    else:
        lines += notice

    return '\n'.join(lines)


def _normalize_docstring(docstring):
    """Normalizes the docstring.

    Replaces tabs with spaces, removes leading and trailing blanks lines, and
    removes any indentation.

    Copied from PEP-257:
    https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation

    Args:
        docstring: the docstring to normalize

    Returns:
        The normalized docstring
    """
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    # (we use sys.maxsize because sys.maxint doesn't exist in Python 3)
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)
