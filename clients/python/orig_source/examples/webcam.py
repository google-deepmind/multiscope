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
