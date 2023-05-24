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

"""Example provides a simple example of triggering callbacks from Multiscope.

WARNING: this API works but is not stable yet.
"""
import time
from typing import Sequence

from absl import app

from golang.multiscope.streams.userinputs import userinputs_pb2
import multiscope
from multiscope.examples import examples

# Dictionary that will be updated on key events.
storage = {
    "keyboard_counter": 0,
}
keyboard_counter = 0
text_writer = None
keyboard_writer = None
mouse_writer = None


def keyboard_callback(path: Sequence[str], event: userinputs_pb2.KeyboardEvent):
  """Called when a key is pressed. Also update the text writer."""
  offset = 0
  if len(event.key) == 1:
    offset = 1 if ord(event.key) % 2 == 0 else -1
  if event.ctrl:
    offset *= 2
  if event.alt:
    offset *= 4
  if event.meta:
    offset *= 8
  if event.shift:
    offset *= 16
  storage["keyboard_counter"] += offset
  keyboard_writer.write(storage)
  text_writer.write(f"{path}\n{event}")


def mouse_callback(path: Sequence[str], event: userinputs_pb2.MouseEvent):
  """Called when the mouse move. Also update the text writer."""
  mouse_writer.write({
      "x": event.position_x,
      "y": event.position_y,
      "tx": event.translation_x,
      "ty": event.translation_y,
  })
  text_writer.write(f"{path}\n{event}")


def gamepad_callback(path: Sequence[str], event: userinputs_pb2.GamepadEvent):
  """Called when the gamepad is used. Updates the text writer."""
  text_writer.write(f"{path}\n{event}")


def main(_):
  multiscope.start_server()

  global text_writer
  text_writer = multiscope.TextWriter("Input window")
  text_writer.write("Select this window and use the " +
                    "keyboard, mouse, or gamepad")

  global keyboard_writer
  keyboard_writer = multiscope.ScalarWriter("Keyboard")
  multiscope.events.register_keyboard_callback(keyboard_callback)  # pytype: disable=wrong-arg-types  # gen-stub-imports

  global mouse_writer
  mouse_writer = multiscope.ScalarWriter("Mouse")
  multiscope.events.register_mouse_callback(mouse_callback)  # pytype: disable=wrong-arg-types  # gen-stub-imports

  multiscope.events.register_gamepad_callback(gamepad_callback)  # pytype: disable=wrong-arg-types  # gen-stub-imports

  for _ in examples.step():
    # This example uses sleep to ensure the main thread isn't CPU-bound as this
    # can starve the handlers of the GIL.
    time.sleep(1)


if __name__ == "__main__":
  app.run(main)
