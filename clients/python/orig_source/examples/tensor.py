"""Write a reference tensor to make sure the display is correct."""

from absl import flags
import numpy as np
from numpy import random

from absl import app
import multiscope
from multiscope.examples import examples

FLAGS = flags.FLAGS


def create_reference_image():
  """Return a reference tensor."""
  dim_y, dim_x = 5, 11
  img = np.zeros((dim_y, dim_x), dtype=np.uint8)
  for x in range(dim_x):
    img[0, x] = 55
    img[dim_y - 1, x] = 155

  img[int(dim_y / 2) - 1][int(dim_x / 2)] = 55
  img[int(dim_y / 2)][int(dim_x / 2)] = 100
  img[int(dim_y / 2) + 1][int(dim_x / 2)] = 155

  img[int(dim_y / 2)][int(dim_x / 4)] = 55
  img[int(dim_y / 2)][int((dim_x / 4) * 3)] = 155

  print(img)
  return img


def create_reference_tensor(img):
  """Take a reference tensor and duplicate that tensor along multiple dimensions."""
  dims = [5, 4, 3, 2]
  tensor = img
  for dim in dims:
    nt = np.zeros(tuple([dim]) + tensor.shape, dtype=np.uint8)
    for i in range(dim):
      nt[i] = tensor
    tensor = nt
  print('Tensor', tensor.shape)
  print(tensor)
  return tensor


def main(_):
  multiscope.start_server()
  ticker = multiscope.Ticker('main')
  tensor2d = create_reference_image()
  image_writer = multiscope.ImageWriter('Reference')
  image_writer.write(tensor2d)
  ref_tensor = create_reference_tensor(tensor2d).astype(np.float32)

  tensor = ref_tensor.copy()
  tensor_writer = multiscope.TensorWriter('Tensor')
  timescale = 5000.0
  for t in examples.step():
    if t % timescale < (timescale / 2):
      tensor += random.uniform(low=-1, high=1, size=tensor.shape)
    else:
      step_size = 10 / timescale
      tensor += step_size * (ref_tensor - tensor)
    tensor_writer.write(tensor)
    ticker.tick()


if __name__ == '__main__':
  app.run(main)
