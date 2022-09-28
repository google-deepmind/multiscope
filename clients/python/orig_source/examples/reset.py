"""Example provides a simple example of using Multiscope."""

import math

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("reset")
  for tick in examples.step():
    w.reset()
    for tick_sub in range(0, 50):
      w.write({
          "tick_sub": math.sin((tick * 100 + tick_sub) * 0.5),
      })


if __name__ == "__main__":
  app.run(main)
