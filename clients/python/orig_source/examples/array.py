"""Example provides a simple example of using Multiscope with arrays."""

import math

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("sin")
  w.set_history_length(400)
  for time in examples.step():
    w.write({
        "sin": [math.sin(time * 0.01),
                math.sin(time * 0.02)],
    })

if __name__ == "__main__":
  app.run(main)
