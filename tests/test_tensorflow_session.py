#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest
import pytest

from tests.base import CustomTestCase


class TensorflowSessionTestCase(CustomTestCase):

    JOB_NAME = "tf_session_example"

    def test_execution(self):
        self.assertTrue(self.run_command(TensorflowSessionTestCase.JOB_NAME))

    @pytest.mark.run(after='test_execution')
    def test_report_is_compliant(self):
        reference_count = -1

        range_names = [
            # ("name", time_target)
            ("Dense 1", 9e4),  # 93,230
            ("Dense 1 grad", 1.9e5),  # 191,680
            ("Dense 2", 9e4),  # 92,629
            ("Dense 2 grad", 1.9e5),  # 193,426
            ("Dense 3", 1e5),  # 103,867
            ("Dense 3 grad", 2.0e5),  # 196,523
            ("Dense 4", 9e4),  # 86,882
            ("Dense 4 grad", 2.0e5),  # 204,076
            ("Dense 5", 5e4),  # 49,689
            ("Dense 5 grad", 1.7e5),  # 172,503
            ("Dense Block", 8.7e5),  # 866,847
            ("Dense Block grad", 1.02e6)  # 1,017,128
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
