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

"""async provides an example for plotting values at different point in time."""

import math

from absl import app
import altair as alt

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("sin")
  w.set_history_length(50)
  w.chart = alt.Chart(
      w.altair_data, width=800, height=400).mark_line(point=True).encode(
          x=alt.X("__time__:Q", axis=alt.Axis(title="Time")),
          y="Value:Q",
          color="Label:N")
  for time in examples.step():
    w.write({
        "sin": math.sin(time * 0.1),
    })
    if time % 10 == 0:
      w.write({
          "__time__": time,
          "cos": math.cos(time * 0.1),
      })


if __name__ == "__main__":
  app.run(main)
