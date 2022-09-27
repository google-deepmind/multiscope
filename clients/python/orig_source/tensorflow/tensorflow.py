"""Module for plotting tensorflow data with multiscope."""
from typing import Any

import numpy as np
import tensorflow.compat.v1 as tf
import tree

from multiscope.remote.writers import base

Nest = Any  # pylint: disable=invalid-name


def write_tf(writer: base.Writer, tf_data: Nest) -> tf.Operation:
  """Creates a TF op that writes Tensors to a given Multiscope writer.

  See examples/tensorflow.py for detailed usage.

  Args:
    writer: A multiscope.Writer to send the data to.
    tf_data: A tf.Tensor containing data to pass to the writer (e.g. image), or
      any nested structure (e.g. dictionary) thereof.

  Returns:
    A TF op which when run will pass data to the writer.
  """
  flat_tf_data = tree.flatten(tf_data)

  def _write_tf(*flat_data):
    data = tree.unflatten_as(tf_data, flat_data)
    writer.write(data)
    return np.zeros(1, np.int32)  # We need to return something.

  out = tf.numpy_function(_write_tf, flat_tf_data, [tf.int32])
  return out[0].op
