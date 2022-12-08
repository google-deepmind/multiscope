"""A simple example of using a TextWriter."""

import time
import numpy as np

from absl import app

import multiscope
from examples import common


def main(argv):
    multiscope.start_server()
    ticker = multiscope.Ticker("ticker")
    text = multiscope.TensorWriter("Random Image", ticker)
    shape = (12, 7, 3)
    for _ in common.step():
        ticker.tick()
        text.write(np.random.randint(low=0, high=255, size=shape, dtype=np.uint8))


if __name__ == "__main__":
    app.run(main)
