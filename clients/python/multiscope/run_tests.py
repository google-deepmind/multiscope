import unittest
from absl import app

from multiscope import test_control
from absl import flags

ONLY_FAST = flags.DEFINE_bool(
    "only_fast_tests",
    False,
    "Run only the fast tests.",
)


def main(argv):
    test_control.only_fast = ONLY_FAST.value
    loader = unittest.TestLoader()
    tests = loader.discover("multiscope", pattern="*_test.py")
    if not tests:
        raise RuntimeError("Could not find any tests!")

    testRunner = unittest.runner.TextTestRunner(verbosity=2)
    testRunner.run(tests)


if __name__ == "__main__":
    app.run(main)
