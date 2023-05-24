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

"""Tests for the example in the reflect package."""

from absl import flags
from absl.testing import flagsaver
import portpicker

from multiscope.go.examples.python import example
from absl.testing import absltest

FLAGS = flags.FLAGS

NSTEPS = 100


class ExampleTest(absltest.TestCase):

  @flagsaver.flagsaver
  def test_example(self):
    FLAGS.step_limit = NSTEPS
    FLAGS.port = portpicker.pick_unused_port()
    example.main(None)


if __name__ == '__main__':
  absltest.main()
