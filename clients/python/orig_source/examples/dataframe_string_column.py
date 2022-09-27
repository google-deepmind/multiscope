"""DataFrame shows how to write dataframe to Multiscope."""

from absl import app
import altair as alt
import numpy as np
import pandas as pd

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  clock = multiscope.Clock("main")
  w = multiscope.DataFrameWriter("Data Frame", clock)
  w.chart = alt.Chart(
      w.altair_data, width=400, height=400).mark_rect().encode(
          x="x:O", y="y:N", color="z:Q")
  for _ in examples.step():
    x, y = np.meshgrid(range(10), range(10))
    data = np.random.uniform(size=(10, 10))
    w.write(
        pd.DataFrame.from_dict({
            "x": x.ravel(),
            "y": [chr(ord("A") + elem) for elem in y.ravel()],
            "z": data.ravel(),
        }))
    clock.tick()


if __name__ == "__main__":
  app.run(main)
