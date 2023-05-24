# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import sys

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
    results = testRunner.run(tests)

    if results.wasSuccessful():
        return sys.exit(0)
    else:
        return sys.exit(1)


if __name__ == "__main__":
    app.run(main)
