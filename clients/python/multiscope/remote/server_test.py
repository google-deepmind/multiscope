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

import time

from absl.testing import absltest

import multiscope
from multiscope import test_control


class TestServer(absltest.TestCase):

    # Users may be starting multiple servers. Although this is a programming
    # error, multiscope should work normally.
    @absltest.skipIf(test_control.only_fast, "Only running fast tests.")
    def testStartServerMultipleTimes(self):
        for _ in range(10):
            multiscope.start_server()
        # start_server is asynchronous, wait for multiscope to attempt startup
        time.sleep(10)


if __name__ == "__main__":
    absltest.main()
