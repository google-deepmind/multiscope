"""Example provides a simple example of using Multiscope."""

import math

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("sin")
  for time in examples.step():
    w.write({
        "sin": math.sin(time),
    })


if __name__ == "__main__":
  app.run(main)
