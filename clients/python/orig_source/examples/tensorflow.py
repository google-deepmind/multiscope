"""Encapsulating writers into Tensorflow ops to visualize tensors."""

from absl import app
import tensorflow.compat.v1 as tf

import multiscope
from multiscope import tensorflow as multiscope_tf
from multiscope.examples import examples


def main(_):
  multiscope.start_server()

  time = tf.get_variable("time", initializer=0.)

  # Image example.
  # Create some dummy image: sin on a 2d grid.
  image_writer = multiscope.ImageWriter("Psychedelic")
  # Create the 2d grid: grid[i, j] = (i + j) / const
  x = tf.range(0, 1, 0.001)
  grid = x[:, None] + x[None, :]
  image = tf.sin(grid + time)
  # Add channels dim.
  image = tf.tile(image[:, :, None], [1, 1, 3])
  # Convert to image format: [0, 255] uint8.
  image = tf.cast(image * 255, tf.uint8)
  # Create the op which on every execution will send the current image to
  # multiscope.
  im_write_op = multiscope_tf.write_tf(image_writer, image)

  # Scalar plot example.
  plot_writer = multiscope.ScalarWriter("sin")
  plot_write_op = multiscope_tf.write_tf(plot_writer, {
      "sin": tf.sin(time),
      "cos": tf.cos(time)
  })

  # Create an op which updates the timer and has the multiscope ops as
  # dependencies.
  with tf.control_dependencies([im_write_op, plot_write_op]):
    time = tf.assign_add(time, 0.01)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in examples.step():
      sess.run(time)


if __name__ == "__main__":
  app.run(main)
