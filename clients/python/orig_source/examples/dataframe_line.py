"""Plots a line from x,y pairs, using DataFrameWriter."""

import collections
from absl import app
import altair as alt
import numpy as np
import pandas as pd

import multiscope
from multiscope.examples import examples


def main(_):
  """Plots the brownian motion trajectory of a 2D point."""

  multiscope.start_server()
  clock = multiscope.Clock("main")
  w = multiscope.DataFrameWriter("Data Frame", clock)
  max_abs_value = 1.0
  value_range = [-max_abs_value, max_abs_value]
  w.chart = alt.Chart(
      w.altair_data, width=400, height=400).mark_line(point=True).encode(
          alt.X("x:Q", scale=alt.Scale(domain=value_range)),
          alt.Y("y:Q", scale=alt.Scale(domain=value_range)),
          order="t:Q")
  size = 20
  ts = collections.deque(maxlen=size)
  xs = collections.deque(maxlen=size)
  ys = collections.deque(maxlen=size)
  xy = np.zeros(2)

  for t in examples.step():
    ts.append(t)
    xs.append(xy[0])
    ys.append(xy[1])
    w.write(pd.DataFrame.from_dict({"t": ts, "x": xs, "y": ys}))
    clock.tick()
    xy += np.random.normal(size=2) * .05

    # Clear trajectory if it goes out of bounds
    if np.any(np.abs(xy) > max_abs_value):
      xy.fill(0.0)
      xs.clear()
      ys.clear()
      ts.clear()


if __name__ == "__main__":
  app.run(main)
