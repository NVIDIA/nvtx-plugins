#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest

from tests.base import CustomTestCase


class TensorflowSessionTestCase(CustomTestCase):

    @pytest.mark.run(order=1)
    def test_execution(self):
        self.assertTrue(self.run_command("tf_session_example"))


if __name__ == '__main__':
    unittest.main()
