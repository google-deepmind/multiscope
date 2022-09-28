"""Example testing the reflect API of Multiscope."""

import math

from absl import flags
import numpy as np

from absl import app
import multiscope
from multiscope.examples import examples

FLAGS = flags.FLAGS


class DataUpdater:
  """Updates some data."""

  def __init__(self):
    self.t = 0
    self.cos_t = math.cos(self.t)
    self.sin_t = math.sin(self.t)
    self.tensor = np.arange(0, 100)

  def step(self):
    self.t = self.t + 1
    self.cos_t = math.cos(self.t * 0.01)
    self.sin_t = math.sin(self.t * 0.01)
    self.tensor = self.t * 0.1 + np.arange(0, 100)
    self.tensor = np.sin((self.tensor * 2 * math.pi) / len(self.tensor))

  @multiscope.reflect_attrs
  def export_attrs(self):
    return multiscope.all_non_callable_attrs(self)


def main(_):
  multiscope.start_server()
  ticker = multiscope.Ticker("main")
  updater = DataUpdater()
  multiscope.reflect(ticker, "updater", updater)
  for _ in examples.step():
    updater.step()
    ticker.tick()


if __name__ == "__main__":
  app.run(main)
