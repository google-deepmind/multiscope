"""nans is an example to test how vega behaves with NaNs."""

import math
import time

from absl import app

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  w = multiscope.ScalarWriter("Sin with NaNs")
  w.set_history_length(400)
  for t in examples.step():
    val = math.sin(t * 0.01)
    if t % 400 < 100:
      val = float("NaN")
    w.write({
        "sin": val,
        "nan": float("NaN"),
    })
    time.sleep(0.001)


if __name__ == "__main__":
  app.run(main)
