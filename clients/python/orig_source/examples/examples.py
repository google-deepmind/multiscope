"""Examples is a module to help to run the examples automatically."""

from typing import Optional, Text

from absl import flags

from multiscope.remote import clock
from multiscope.remote import group

flags.DEFINE_integer("step_limit", None, "Limit on the number of steps.")

FLAGS = flags.FLAGS


def step():
  """step iterates until a maximum number of steps has been reached."""
  time = 0
  while FLAGS.step_limit is None or time < FLAGS.step_limit:
    yield time
    time = time + 1


class Ticker(clock.Ticker):
  """Ticker which returns False when the number of steps has been set and is exhausted."""

  def __init__(self, name: Text, parent: Optional[group.ParentNode] = None):
    super().__init__(name, parent)
    self.time = 0

  def tick(self) -> bool:
    super().tick()
    self.time += 1
    return FLAGS.step_limit is None or self.time < FLAGS.step_limit
