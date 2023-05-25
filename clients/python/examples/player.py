"""A simple example of using a Player."""

import time

from absl import app
import multiscope
from examples import common


def main(argv):
  multiscope.start_server()
  player = multiscope.Player("player")
  text = multiscope.TextWriter("Current time", player)
  for _ in common.step():
    player.store_frame()
    text.write(time.strftime("%H:%M:%S %Z on %b %d, %Y"))


if __name__ == "__main__":
  app.run(main)
