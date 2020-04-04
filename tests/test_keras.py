#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest

from tests.base import CustomTestCase


class KerasTestCase(CustomTestCase):

    JOB_NAME = "keras_example"

    def test_execution(self):
        self.assertTrue(self.run_command(KerasTestCase.JOB_NAME))

    @pytest.mark.run(after='test_execution')
    def test_report_is_compliant(self):
        reference_count = -1

        range_names = [
            # ("name", time_target)
            ("Dense 1", 2.4e5),  # 243,147
            ("Dense 1 grad", 2.3e5),  # 228,901
            ("Dense 2", 9e4),  # 90,982
            ("Dense 2 grad", 3.6e5),  # 360,840
            ("Dense 3", 1.0e5),  # 97,494
            ("Dense 3 grad", 1.7e5),  # 170,699
            ("Dense 4", 7e4),  # 67,857
            ("Dense 4 grad", 1.8e5),  # 181,303
            ("Dense 5", 7e4),  # 66,138
            ("Dense 5 grad", 1.7e5)  # 173,618
        ]

        with self.open_db(KerasTestCase.JOB_NAME) as conn:

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
