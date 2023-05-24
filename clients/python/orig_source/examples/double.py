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

"""Double is an example illustrating how to use two writers."""

import math

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  wsin = multiscope.ScalarWriter("sin")
  wsin.set_history_length(400)
  wcos = multiscope.ScalarWriter("cos")
  wcos.set_history_length(400)
  for time in examples.step():
    wsin.write({
        "sin": math.sin(time * 0.01),
    })
    wcos.write({
        "cos": math.cos(time * 0.01),
    })


if __name__ == "__main__":
  app.run(main)
