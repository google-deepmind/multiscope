"""A simple example of using a Ticker."""

import time

from absl import app
import multiscope
from examples import common


def main(argv):
  multiscope.start_server()
  ticker = multiscope.Ticker("ticker")
  text = multiscope.TextWriter("Current time", ticker)
  for _ in common.step():
    ticker.tick()
    text.write(time.strftime("%H:%M:%S %Z on %b %d, %Y"))


if __name__ == "__main__":
  app.run(main)
