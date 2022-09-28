"""spec shows how to specify a vega spec to a writer."""

import math

from absl import app
import altair as alt

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("sincos")
  w.set_history_length(1)
  w.chart = alt.Chart(
      w.altair_data, width=400, height=400).mark_bar().encode(
          alt.X("Label:O"), alt.Y("Value:Q", scale=alt.Scale(domain=[-1, 1])))
  for time in examples.step():
    w.write({
        "sin": math.sin(time * 0.0001),
        "cos": math.cos(time * 0.0001),
    })


if __name__ == "__main__":
  app.run(main)
