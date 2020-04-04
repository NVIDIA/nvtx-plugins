import pytest

SUCCESS_CODE = 0
SIGKILL_CODE = 9
SIGTERM_CODE = 15

# def test_nsys():
#     import subprocess
#
#     # cmd = "nsys profile -d 60 -w true --force-overwrite=true --sample=cpu -t 'nvtx,cuda' --stop-on-exit=true --kill=sigkill -o examples/tf_session_example python examples/tf_session_example.py"
#
#     modified_command = [
#         'nsys',
#         'profile',
#         # '--delay=10',
#         '--duration=30',
#         '--sample=cpu',
#         '--trace=nvtx,cuda',
#         '--output=examples/tf_session_example',
#         '--force-overwrite=true',
#         '--export=sqlite',
#         '--stop-on-exit=true',
#         '--kill=sigkill'
#     ]
#
#     py_command = "python examples/tf_session_example.py"
#     run_command = modified_command + py_command.split(" ")
#
#     proc = subprocess.run(" ".join(run_command), shell=True, check=False)
#
#     if proc.returncode not in [SUCCESS_CODE, SIGKILL_CODE, SIGTERM_CODE]:
#         raise RuntimeError("Return Code: %d" % proc.returncode)
