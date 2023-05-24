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

"""Example provides a simple example of using Multiscope clocks."""

import math

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  clock = multiscope.Clock("main")
  w = multiscope.ScalarWriter("sin1", clock)
  t = multiscope.TextWriter("sin2", clock)

  # set an initial period, in milliseconds
  # can then be manually updated from the frontend
  clock.period = 500

  for time in examples.step():
    w.write({
        "sin": math.sin(time),
    })
    t.write("The value of sin(%d) is %f" % (time, math.sin(time)))
    clock.tick()


if __name__ == "__main__":
  app.run(main)
