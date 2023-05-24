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

"""Example testing the reflect API of Multiscope."""

import math

from absl import flags
import numpy as np

from absl import app
import multiscope
from multiscope.examples import examples

FLAGS = flags.FLAGS


class DataUpdater:
  """Updates some data."""

  def __init__(self):
    self.t = 0
    self.cos_t = math.cos(self.t)
    self.sin_t = math.sin(self.t)
    self.tensor = np.arange(0, 100)

  def step(self):
    self.t = self.t + 1
    self.cos_t = math.cos(self.t * 0.01)
    self.sin_t = math.sin(self.t * 0.01)
    self.tensor = self.t * 0.1 + np.arange(0, 100)
    self.tensor = np.sin((self.tensor * 2 * math.pi) / len(self.tensor))

  @multiscope.reflect_attrs
  def export_attrs(self):
    return multiscope.all_non_callable_attrs(self)


def main(_):
  multiscope.start_server()
  ticker = multiscope.Ticker("main")
  updater = DataUpdater()
  multiscope.reflect(ticker, "updater", updater)
  for _ in examples.step():
    updater.step()
    ticker.tick()


if __name__ == "__main__":
  app.run(main)
