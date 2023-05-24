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

"""image provides an example for streaming a greyscale image."""

from absl import app
import numpy as np
from six.moves import range

import multiscope
from multiscope.examples import examples


class Board(object):
  """Board class that scans a rectangle top left -> bottom right (wrapping).

  A NumPy array like np.ones((10, 5))  has 10 rows and 5 columns.
  If you print it, it will be 10 rows tall, and 5 columns wide.
  It will print downwards, so [0, 0] will be the top left.

  This is like np.ones((height, width)).
  If trying to index as Cartesian coordinates, it is my_array[y, x].

  This can be confusing.

  This example is meant to demonstrate top/bottom left/right.
  """

  def __init__(self, width, height):
    super(Board, self).__init__()
    self.width = width
    self.height = height
    self.pos_x = self.pos_y = 0
    # 2D array will be rendered as grayscale. All values should be in [0, 255].
    self.img = np.zeros([self.height, self.width], dtype=np.uint8)

  def step(self):
    """Step makes the pixel move one step."""
    self.pos_x += 1
    if self.pos_x == self.width:
      self.pos_x = 0
      self.pos_y += 1
      if self.pos_y == self.height:
        self.pos_y = 0
    self.img.fill(0)
    self.img[self.pos_y, self.pos_x] = 255


class Box(object):
  """Box class containing a square ball."""

  def __init__(self, size_y, size_x, ball_size):
    super(Box, self).__init__()
    self.size_x = size_x
    self.size_y = size_y
    self.pos_x = size_x / 2
    self.pos_y = size_y / 2
    self.vel_x = 3
    self.vel_y = 5
    self.ball_size = ball_size
    self.img = np.zeros([size_y, size_x], dtype=np.uint8)

  def step(self):
    """Step makes the box move one step."""
    self.pos_x += self.vel_x
    if self.pos_x < 0:
      self.pos_x = 0
      self.vel_x *= -1
    if self.pos_x + self.ball_size > self.size_x - 1:
      self.pos_x = self.size_x - self.ball_size - 1
      self.vel_x *= -1

    self.pos_y += self.vel_y
    if self.pos_y < 0:
      self.pos_y = 0
      self.vel_y *= -1
    if self.pos_y + self.ball_size > self.size_y - 1:
      self.pos_y = self.size_y - self.ball_size - 1
      self.vel_y *= -1

    self.img.fill(0)
    for ix in range(self.ball_size):
      for iy in range(self.ball_size):
        self.img[int(self.pos_y + iy), int(self.pos_x + ix)] = 255


def main(_):
  multiscope.start_server()
  box = Box(100, 200, 10)
  board = Board(width=5, height=10)
  clock = multiscope.Clock("main")
  board_writer = multiscope.ImageWriter("Scanner", clock)
  box_writer = multiscope.ImageWriter("Box", clock)
  for _ in examples.step():
    box_writer.write(box.img)
    board_writer.write(board.img)
    box.step()
    board.step()
    clock.tick()


if __name__ == "__main__":
  app.run(main)
