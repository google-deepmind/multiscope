# Copyright 2023 DeepMind Technologies Limited
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
"""A simple example of using a TextWriter."""

import time

from absl import app
from examples import common
import multiscope


text_writer = None
keyboard_writer = None
mouse_writer = None


def keyboard_callback(event: multiscope.KeyboardEvent):
  """Called when a key is pressed. Also update the text writer."""
  offset = event.key
  if event.ctrl:
    offset *= 2
  if event.alt:
    offset *= 4
  if event.meta:
    offset *= 8
  if event.shift:
    offset *= 16
  keyboard_writer.write({
      'modified key': offset,
  })
  text_writer.write(f"{event}")


def mouse_callback(event: multiscope.MouseEvent):
  """Called when the mouse move. Also update the text writer."""
  mouse_writer.write({
      "x": event.position_x,
      "y": event.position_y,
      "tx": event.translation_x,
      "ty": event.translation_y,
  })
  text_writer.write(f"{event}")


def main(_):
  multiscope.start_server()

  global text_writer
  text_writer = multiscope.TextWriter("Events")

  global keyboard_writer
  keyboard_writer = multiscope.ScalarWriter("Keyboard")
  multiscope.register_keyboard_callback(keyboard_callback)

  global mouse_writer
  mouse_writer = multiscope.ScalarWriter("Mouse")
  multiscope.register_mouse_callback(mouse_callback)

  for _ in common.step():
    # This example uses sleep to ensure the main thread isn't CPU-bound as this
    # can starve the handlers of the GIL.
    time.sleep(1)


if __name__ == "__main__":
  app.run(main)
