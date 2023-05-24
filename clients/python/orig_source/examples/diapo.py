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

"""image_rgb provides an example for streaming an RGB image."""

import os
import time

from absl import app
import numpy as np
from skimage.io import imread

import multiscope


def rgb2gray(rgb):
  r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray.astype(np.uint8)


def main(_):
  multiscope.start_server()
  # TODO: the path assumes that this example is run from the main multiscope  # pylint: disable=g-bad-todo
  # directory. Generalize.
  img_path = "./examples/images/"
  imgs = []
  for img_file in os.listdir(img_path):
    if not img_file.endswith(".jpg"):
      continue
    img = imread(img_path + img_file)
    imgs.append(img)
    imgs.append(rgb2gray(img))
  w = multiscope.ImageWriter("diapo")
  while True:
    for img in imgs:
      w.write(img)
      time.sleep(1)


if __name__ == "__main__":
  app.run(main)
