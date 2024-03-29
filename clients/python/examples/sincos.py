# Copyright 2023 DeepMind Technologies Limited
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
"""sincos provides an example for plotting two time-series in one chart."""

import math

from absl import app
import multiscope
from examples import common


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("sincos")
  for time in common.step():
    time = time * 0.002
    w.write({
        "sin": math.sin(time),
        "cos": math.cos(time),
    })


if __name__ == "__main__":
  app.run(main)
