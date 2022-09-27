"""sincos provides an example for plotting two time-series in one chart."""

import math

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("sin")
  for time in examples.step():
    time = time + 0.01
    w.write({
        "sin": math.sin(time),
        "cos": math.cos(time),
    })


if __name__ == "__main__":
  app.run(main)
