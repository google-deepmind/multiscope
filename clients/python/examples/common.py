# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
