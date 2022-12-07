"""A simple example of using a TextWriter."""

import time

from absl import app
import multiscope
from examples import common


def main(argv):
    multiscope.start_server()
    text = multiscope.TextWriter("Current time")
    for _ in common.step():
        text.write(time.strftime("%H:%M:%S %Z on %b %d, %Y"))


if __name__ == "__main__":
    app.run(main)
