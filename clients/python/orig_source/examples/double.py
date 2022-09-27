"""Double is an example illustrating how to use two writers."""

import math

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  wsin = multiscope.ScalarWriter("sin")
  wsin.set_history_length(400)
  wcos = multiscope.ScalarWriter("cos")
  wcos.set_history_length(400)
  for time in examples.step():
    wsin.write({
        "sin": math.sin(time * 0.01),
    })
    wcos.write({
        "cos": math.cos(time * 0.01),
    })


if __name__ == "__main__":
  app.run(main)
