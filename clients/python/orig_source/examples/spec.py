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

"""spec shows how to specify a vega spec to a writer."""

import math

from absl import app

import multiscope
from multiscope.examples import examples


chart_spec = r"""
{
  "$schema": "https://vega.github.io/schema/vega-lite/v2.json",
  "height": 400,
  "width": 400,
  "mark": "bar",
  "encoding": {
    "x": {"field": "Label", "type": "ordinal"},
    "y": {"field": "Value", "type": "quantitative", "scale": {"domain": [-1,1]}}
  }
}
"""


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("sincos")
  w.set_history_length(1)
  w.set_spec(chart_spec)
  for time in examples.step():
    w.write({
        "sin": math.sin(time * 0.0001),
        "cos": math.cos(time * 0.0001),
    })


if __name__ == "__main__":
  app.run(main)
