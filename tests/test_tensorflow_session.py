#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest

from tests.base import CustomTestCase


class TensorflowSessionTestCase(CustomTestCase):

    JOB_NAME = "tf_session_example"

    def test_execution(self):
        self.assertTrue(self.run_command(TensorflowSessionTestCase.JOB_NAME))

    @pytest.mark.run(after='test_execution')
    def test_report_is_compliant(self):

        import nvtx
        import sys
        # ['/usr/local/lib/python3.6/dist-packages/nvtx']
        print(nvtx.__path__._path, file=sys.stderr)

        reference_count = -1

        range_names = [
            "Dense 1",
            "Dense 1 grad",
            "Dense 2",
            "Dense 2 grad",
            "Dense 3",
            "Dense 3 grad",
            "Dense 4",
            "Dense 4 grad",
            "Dense 5",
            "Dense 5 grad",
            "Dense Block",
            "Dense Block grad"
        ]

        with self.open_db(TensorflowSessionTestCase.JOB_NAME) as conn:

            for range in range_names:

                try:
                    count, avg_exec_time = self.query_report(
                        conn,
                        range_name=range
                    )

                    if reference_count < 0:
                        # At least 500 steps should be processed
                        self.assertGreater(count, 500)
                        reference_count = count
                        continue

                    # The profile could start & end in the middle of one step.
                    # Hence the a we check for a range instead of strict equal.
                    self.assertGreaterEqual(count, reference_count - 1)
                    self.assertLessEqual(count, reference_count + 1)

                except AssertionError as e:
                    raise AssertionError("Issue with range: %s" % range) from e

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
