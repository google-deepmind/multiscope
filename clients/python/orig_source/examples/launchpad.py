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

"""This example shows how to use multiscope with launchpad.

In this example, each node runs its own multiscope. To access the dashboard for
a particular node see its logs.

Use the the flag `--lp_launch_type=[local_mp|xmanager]` to launch locally or
on xmanager, respectively.
"""

import time
from absl import app
import launchpad as lp
import multiscope


class CurrentTimeNode:
  """Example node that writes the current time to multiscope."""

  def __init__(self):
    """Initializes the launchpad node."""
    # Each node starts its own multiscope server.
    multiscope.start_server()
    self._text_writer = multiscope.TextWriter('Current time')

  def run(self):
    """Entrypoint for the launchpad node."""
    while True:
      self.step()

  def step(self):
    self._text_writer.write(time.strftime('%H:%M:%S %Z on %b %d, %Y'))


def make_program(num_nodes: int = 2) -> lp.Program:
  """Define the distributed program topology."""
  program = lp.Program('launchpad_multiscope_example')

  with program.group('current_time'):
    for _ in range(num_nodes):
      program.add_node(lp.CourierNode(CurrentTimeNode))

  return program


def main(_):
  program = make_program()
  lp.launch(
      program,
      # TODO: potentially manage the multiscope port on xmanager.  # pylint: disable=g-bad-todo
      # Set `multi_port=0` when running locally to ensure unused ports are
      # picked for each node's multiscope instance, otherwise they may clash
      # when racing to pick the default port.
      local_resources=dict(
          current_time=lp.PythonProcess(args=dict(multi_port=0))))


if __name__ == '__main__':
  app.run(main)
