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
