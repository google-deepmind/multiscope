"""sincos provides an example for plotting two time-series in one chart."""

import math

from absl import app
import multiscope
from examples import common


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("sincos")
  for time in common.step():
    time = time * 0.002
    w.write({
        "sin": math.sin(time),
        "cos": math.cos(time),
    })


if __name__ == "__main__":
  app.run(main)
