#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import time
import unittest


__all__ = [
    'CustomTestCase',
]

SUCCESS_CODE = 0
SIGKILL_CODE = 9


class CustomTestCase(unittest.TestCase):

    def run_command(self, job_name):

        for ext in ["qdrep", "sqlite"]:
            try:
                os.remove("examples/%s.%s" % (job_name, ext))
            except FileNotFoundError:
                pass

        def exec_cmd(cmd):
            command_proc = subprocess.Popen(cmd)
            return_code = command_proc.wait()

            if return_code not in [SUCCESS_CODE, SIGKILL_CODE]:
                sys.tracebacklimit = 0
                stdout, stderr = command_proc.communicate()
                raise RuntimeError(
                    "\n##################################################\n"
                    "[*] STDOUT:{error_stdout}\n"
                    "[*] STERR:{error_stderr}\n"
                    "[*] command launched: `{command}`\n"
                    "##################################################\n".format(
                        error_stdout=stdout.decode("utf-8"),
                        error_stderr=stderr.decode("utf-8"),
                        command=" ".join(cmd)
                    )
                )

            return True

        modified_command = [
            'nsys',
            'profile',
            '--delay=10',
            '--duration=30',
            '--sample=cpu',
            '--trace=nvtx,cuda',
            '--output=examples/%s' % job_name,
            '--force-overwrite=true',
            '--stop-on-exit=true',
            '--kill=sigkill'
        ]

        py_command = "python examples/%s.py" % job_name
        run_command = modified_command + py_command.split(" ")

        print("Command Executed: %s" % (" ".join(run_command)), file=sys.stderr)

        self.assertTrue(exec_cmd(run_command))

        base_path = "examples/%s." % job_name
        self.assertTrue(os.path.exists(base_path + "qdrep"))

        command_export = [
            "nsys-exporter",
            "--export-sqlite",
            "--input-file=examples/%s.qdrep" % job_name,
            "--output-file=examples/%s.sqlite" % job_name
        ]

        self.assertTrue(exec_cmd(command_export))
        self.assertTrue(os.path.exists(base_path + "sqlite"))

        return True
