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

"""Multiscope example for visualizing gamepad state."""

from absl import logging

from absl import app

import multiscope
from multiscope.contrib.py.inputs import gamepad_viz
from multiscope.examples import examples


def main(argv):
  del argv  # Unused by main.

  logging.info("Starting Multiscope server.")
  multiscope.start_server()
  clock = multiscope.Clock("main")
  clock.period = 500

  gamepad_demo = gamepad_viz.GamepadViz()  # pylint: disable=unused-variable

  for _ in examples.step():
    clock.tick()


if __name__ == "__main__":
  app.run(main)
