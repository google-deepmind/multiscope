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
