"""DataFrame shows how to write dataframe to Multiscope."""

from absl import app
import altair as alt
import numpy as np
import pandas as pd

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  w = multiscope.DataFrameWriter("Data Frame")
  w.chart = alt.Chart(
      w.altair_data, width=400, height=400).mark_rect().encode(
          x=alt.X("x:O", axis=alt.Axis(format=".2f")),
          y=alt.Y("y:O", axis=alt.Axis(format=".2f")),
          color="z:Q")
  for _ in examples.step():
    x, y = np.meshgrid(range(10), range(10))
    data = np.random.uniform(size=(10, 10))
    w.write(
        pd.DataFrame.from_dict({
            "x": x.ravel() * 0.12345,
            "y": y.ravel() * 0.6789,
            "z": data.ravel(),
        }))


if __name__ == "__main__":
  app.run(main)
