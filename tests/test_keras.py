#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest

from tests.base import NVTXBaseTest
from distutils.version import LooseVersion

import tensorflow as tf

TIMING_THRESHOLD = 5


class KerasTestCase(NVTXBaseTest):

    JOB_NAME = "keras_example"

    def test_execution(self):
        self.assertTrue(self.run_command(KerasTestCase.JOB_NAME))

    @pytest.mark.run(after='test_execution')
    def test_report_is_compliant(self):
        reference_count = -1

        if LooseVersion(tf.__version__) >= LooseVersion("2.0.0"):
            range_names = [
                # ("name", time_target)
                ("Dense 1", 1.2e5),  # 120,996
                ("Dense 1 grad", 1.2e5),  # 115,999
                ("Dense 2", 6e4),  # 58,643
                ("Dense 2 grad", 9e4),  # 90,279
                ("Dense 3", 8e4),  # 80,368
                ("Dense 3 grad", 1.1e5),  # 110,181
                ("Dense 4", 5e4),  # 54,668
                ("Dense 4 grad", 1.0e5),  # 95,431
                ("Dense 5", 5e4),  # 51,937
                ("Dense 5 grad", 1.8e5)  # 175,346
            ]

        else:
            range_names = [
                # ("name", time_target)
                ("Dense 1", 2.7e5),  # 273,685
                ("Dense 1 grad", 3.5e5),  # 347,366
                ("Dense 2", 1.6e5),  # 161,556
                ("Dense 2 grad", 3.0e5),  # 297,473
                ("Dense 3", 1.5e5),  # 152,980
                ("Dense 3 grad", 3.2e5),  # 315,244
                ("Dense 4", 1.5e5),  # 150,286
                ("Dense 4 grad", 3.3e5),  # 332,801
                ("Dense 5", 7e4),  # 67,789
                ("Dense 5 grad", 1.9e5)  # 185,972
            ]

        with self.open_db(KerasTestCase.JOB_NAME) as conn:

            for range_name, time_target in range_names:

                with self.catch_assert_error(range_name):

                    count, avg_exec_time = self.query_report(
                        conn,
                        range_name=range_name
                    )

                    self.assertGreaterEqual(
                        avg_exec_time, time_target / TIMING_THRESHOLD
                    )
                    self.assertLessEqual(
                        avg_exec_time, time_target * TIMING_THRESHOLD
                    )

                    if reference_count < 0:
                        # At least 500 steps should be processed
                        self.assertGreater(count, 500)
                        reference_count = count

                    # The profile could start & end in the middle of one step.
                    # Hence the a we check for a range instead of strict equal.
                    self.assertGreaterEqual(count, reference_count - 1)
                    self.assertLessEqual(count, reference_count + 1)

            count, _ = self.query_report(
                conn,
                range_name="Train",
                filter_negative_start=False
            )
            self.assertEqual(count, 1)

            count, _ = self.query_report(
                conn,
                range_name="epoch 0",
                filter_negative_start=False
            )
            self.assertEqual(count, 1)

            count, _ = self.query_report(conn, range_name="batch %")
            self.assertGreaterEqual(count, reference_count - 1)
            self.assertLessEqual(count, reference_count + 1)


if __name__ == '__main__':
    unittest.main()
