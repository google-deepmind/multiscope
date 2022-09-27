"""Example provides a simple example of using Multiscope clocks."""

import math

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  clock = multiscope.Clock("main")
  w = multiscope.ScalarWriter("sin1", clock)
  t = multiscope.TextWriter("sin2", clock)

  # set an initial period, in milliseconds
  # can then be manually updated from the frontend
  clock.period = 500

  for time in examples.step():
    w.write({
        "sin": math.sin(time),
    })
    t.write("The value of sin(%d) is %f" % (time, math.sin(time)))
    clock.tick()


if __name__ == "__main__":
  app.run(main)
