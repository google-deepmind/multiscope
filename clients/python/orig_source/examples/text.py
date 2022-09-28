"""Example provides a simple example of using Multiscope."""

import time

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  text = multiscope.TextWriter("Current time")
  for _ in examples.step():
    text.write(time.strftime("%H:%M:%S %Z on %b %d, %Y"))


if __name__ == "__main__":
  app.run(main)
