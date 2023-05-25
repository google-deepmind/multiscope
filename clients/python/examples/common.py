"""This is a module to help to run the examples automatically."""

from typing import Optional, Text

from absl import flags

# from multiscope.remote import clock
# from remote import group

_STEP_LIMIT = flags.DEFINE_integer("step_limit", None,
                                   "Limit on the number of steps.")


def step():
  """step iterates until a maximum number of steps has been reached."""
  time = 0
  while (_STEP_LIMIT.value is None or time < _STEP_LIMIT.value  # pytype: disable=unsupported-operands
        ):
    yield time
    time = time + 1


# # TODO: untested.
# class Ticker(clock.Ticker):
#   """Ticker which returns False when the number of steps has been set and is exhausted."""

#   def __init__(self, name: Text, parent: Optional[group.ParentNode] = None):
#     super().__init__(name, parent)
#     self.time = 0

#   def tick(self) -> bool:
#     super().tick()
#     self.time += 1
#     return _STEP_LIMIT.value is None or self.time < _STEP_LIMIT.value
