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

"""Example showing how to get webcam input."""

from absl import app
import numpy as np

import multiscope
from multiscope.examples import examples


def main(_):
  multiscope.start_server()
  webcam_writer = multiscope.Webcam('Live stream')
  ticker = multiscope.Ticker('ticker')
  image_writer = multiscope.ImageWriter('Webcam Playback')
  # TODO: The webcam panel starts cropping the canvas image if this  # pylint: disable=g-bad-todo
  # (width x height) surpasses (300x150).
  webcam_writer.set_input_size(max_height_pixels=144, max_width_pixels=256)
  webcam_writer.set_frame_rate(fps=15)
  webcam_writer.register_frame_callback(
      lambda frame, _: image_writer.write(np.array(frame)))

  for _ in examples.step():
    ticker.tick()


if __name__ == '__main__':
  app.run(main)
