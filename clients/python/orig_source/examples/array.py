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

"""Example provides a simple example of using Multiscope with arrays."""

import math

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("sin")
  w.set_history_length(400)
  for time in examples.step():
    w.write({
        "sin": [math.sin(time * 0.01),
                math.sin(time * 0.02)],
    })

if __name__ == "__main__":
  app.run(main)
