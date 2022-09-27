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
