#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest

from tests.base import NVTXBaseTest


class TensorflowSessionTestCase(NVTXBaseTest):

    JOB_NAME = "tf_session_example"

    def test_execution(self):
        self.assertTrue(self.run_command(TensorflowSessionTestCase.JOB_NAME))

    @pytest.mark.run(after='test_execution')
    def test_report_is_compliant(self):
        reference_count = -1

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

        with self.open_db(TensorflowSessionTestCase.JOB_NAME) as conn:

            for range_name, time_target in range_names:

                with self.catch_assert_error(range_name):
                    count, avg_exec_time = self.query_report(
                        conn,
                        range_name=range_name
                    )

                    self.assertGreaterEqual(avg_exec_time, time_target / 3)
                    self.assertLessEqual(avg_exec_time, time_target * 3)

                    if reference_count < 0:
                        # At least 500 steps should be processed
                        self.assertGreater(count, 500)
                        reference_count = count
                        continue

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

            count, _ = self.query_report(conn, range_name="step %")
            self.assertGreaterEqual(count, reference_count - 1)
            self.assertLessEqual(count, reference_count + 1)


if __name__ == '__main__':
    unittest.main()
