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

import numpy as np

import multiscope
from absl.testing import absltest


class DataUpdater02:

  def __init__(self, updater01):
    self.updater01 = updater01

  @multiscope.reflect_attrs
  def export_attrs(self):
    return multiscope.all_non_callable_attrs(self)


class DataUpdater01:
  """Updates some data."""

  def __init__(self):
    self.scalar = 42
    self.self_reference = self
    self.child_reference = DataUpdater02(self)
    self.tensor = np.arange(0, 100)

  @multiscope.reflect_attrs
  def export_attrs(self):
    return multiscope.all_non_callable_attrs(self)


class CycleTest(absltest.TestCase):

  def test_when_cycle(self):
    multiscope.start_server()
    ticker = multiscope.Ticker("main")
    updater = DataUpdater01()
    # Just test that this function does not fill the stack.
    multiscope.reflect(ticker, "updater", updater)


if __name__ == "__main__":
  absltest.main()
