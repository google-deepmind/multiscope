import unittest
from absl import app


def main(argv):
    loader = unittest.TestLoader()
    tests = loader.discover("multiscope", pattern="*_test.py")
    if not tests:
        raise RuntimeError("Could not find any tests!")

    testRunner = unittest.runner.TextTestRunner(verbosity=2)
    testRunner.run(tests)


if __name__ == "__main__":
    app.run(main)
