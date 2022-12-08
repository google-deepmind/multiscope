"""Controls the level of testing we want to do."""

# How to use:
#
# If you have a slow test, add the
# @absltest.skipIf(not test_control.only_fast, "Only running fast tests.")
# decorator on it. See `multiscope/remote/server_test.py` for an example.
#
# When running a test suite, if you only want to run fast tests, change this
# value to `True` before gathering the tests. See `multiscope/run_tests.py` for
# an example.
only_fast = False
