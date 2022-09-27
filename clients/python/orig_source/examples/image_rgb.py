"""image_rgb provides an example for streaming an RGB image."""

import math

from absl import app
import numpy as np
from six.moves import range

import multiscope
from multiscope.examples import examples

benchmark_images = {
    'WhiteUint3Color': np.ones((10, 10, 3), dtype=np.uint8) * 255,
    'WhiteUintGrayScale': np.ones((10, 10), dtype=np.uint8) * 255,
    'BlackUint3Color': np.zeros((10, 10, 3), dtype=np.uint8),
    'BlackUintGrayScale': np.zeros((10, 10), dtype=np.uint8),
    'RedUint3Color': np.ones((10, 10, 3), dtype=np.uint8) * (255, 0, 0),
    'GreenUint3Color': np.ones((10, 10, 3), dtype=np.uint8) * (0, 255, 0),
    'BlueUint3Color': np.ones((10, 10, 3), dtype=np.uint8) * (0, 0, 255),
}


def create_benchmark_images():
  for name in benchmark_images:
    multiscope.ImageWriter(name).write(benchmark_images[name].astype(np.uint8))


def main(_):
  multiscope.start_server()
  psychedelic_writer = multiscope.ImageWriter('Psychedelic')
  create_benchmark_images()
  size_x, size_y = 200, 400
  img = np.zeros([size_y, size_x, 3], dtype=np.uint8)

  for time in examples.step():
    img.fill(0)
    for ic in range(3):
      for ix in range(size_x):
        for iy in range(size_y):
          img[iy, ix, ic] = math.sin(
              (time * (ic + 1) + ix * 0.9 + iy) * 0.02) * 255
    psychedelic_writer.write(img)


if __name__ == '__main__':
  app.run(main)
