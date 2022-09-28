"""Example provides a simple example of using Multiscope."""
import math

from absl import app

import multiscope
from multiscope import mjax
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  writer = mjax.NestedWriter("name")
  for time in examples.step():
    writer.write({"child_a": {"sin1": {"sin": math.sin(time),}}})


if __name__ == "__main__":
  app.run(main)
