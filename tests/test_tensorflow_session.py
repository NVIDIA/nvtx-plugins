#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest

from tests.base import NVTXBaseTest
from distutils.version import LooseVersion

import tensorflow as tf

TIMING_THRESHOLD = 5


class TensorflowSessionTestCase(NVTXBaseTest):

    JOB_NAME = "tf_session_example"

    def test_execution(self):
        self.assertTrue(self.run_command(TensorflowSessionTestCase.JOB_NAME))

    @pytest.mark.run(after='test_execution')
    def test_report_is_compliant(self):
        reference_count = -1

        if LooseVersion(tf.__version__) >= LooseVersion("2.0.0"):
            range_names = [
                # ("name", time_target)
                ("Dense 1", 1.0e5),  # 98,144
                ("Dense 1 grad", 1.8e5),  # 180,291
                ("Dense 2", 9e4),  # 86,186
                ("Dense 2 grad", 1.9e5),  # 185,389
                ("Dense 3", 7e4),  # 72,238
                ("Dense 3 grad", 1.6e5),  # 156,226
                ("Dense 4", 5e4),  # 54,122
                ("Dense 4 grad", 1.6e5),  # 158,607
                ("Dense 5", 4e4),  # 39,480
                ("Dense 5 grad", 1.7e5),  # 165,014
                ("Dense Block", 4.1e5),  # 408,626
                ("Dense Block grad", 8.8e5)  # 884,339
            ]

        else:
            range_names = [
                # ("name", time_target)
                ("Dense 1", 1.3e5),  # 127,042
                ("Dense 1 grad", 3.0e5),  # 304,562
                ("Dense 2", 1.1e5),  # 108,889
                ("Dense 2 grad", 2.2e5),  # 215,664
                ("Dense 3", 1.0e5),  # 101,554
                ("Dense 3 grad", 2.4e5),  # 236,500
                ("Dense 4", 1.0e5),  # 99,070
                ("Dense 4 grad", 2.3e5),  # 225,912
                ("Dense 5", 5e4),  # 47,974
                ("Dense 5 grad", 1.8e5),  # 184,018
                ("Dense Block", 5.7e5),  # 565,927
                ("Dense Block grad", 1.21e6)  # 1,210,117
            ]

        with self.open_db(TensorflowSessionTestCase.JOB_NAME) as conn:

            forward_category_id = self.get_category_id(conn, "Forward")
            backward_category_id = self.get_category_id(conn, "Gradient")

            forward_block_category_id = self.get_category_id(
                conn, "Trace_Forward"
            )
            backward_block_category_id = self.get_category_id(
                conn, "Trace_Gradient"
            )

            for range_name, time_target in range_names:

                if "block" in range_name.lower():
                    category_id_target = (
                        forward_block_category_id
                        if "grad" not in range_name else
                        backward_block_category_id
                    )
                else:
                    category_id_target = (
                        forward_category_id
                        if "grad" not in range_name else
                        backward_category_id
                    )

                with self.catch_assert_error(range_name):
                    count, avg_exec_time, category_id = self.query_report(
                        conn,
                        range_name=range_name
                    )

                    self.assertGreaterEqual(
                        avg_exec_time, time_target / TIMING_THRESHOLD
                    )

                    self.assertLessEqual(
                        avg_exec_time, time_target * TIMING_THRESHOLD
                    )

                    self.assertEqual(category_id, category_id_target)

                    if reference_count < 0:
                        # At least 500 steps should be processed
                        self.assertGreater(count, 500)
                        reference_count = count
                        continue

                    # The profile could start & end in the middle of one step.
                    # Hence the a we check for a range instead of strict equal.
                    self.assertGreaterEqual(count, reference_count - 1)
                    self.assertLessEqual(count, reference_count + 1)

            count, _, category_id = self.query_report(
                conn,
                range_name="Train",
                filter_negative_start=False
            )
            self.assertEqual(count, 1)
            self.assertIsNone(category_id)

            count, _, category_id = self.query_report(conn, range_name="step %")
            self.assertGreaterEqual(count, reference_count - 1)
            self.assertLessEqual(count, reference_count + 1)
            self.assertIsNone(category_id)


if __name__ == '__main__':
    unittest.main()
