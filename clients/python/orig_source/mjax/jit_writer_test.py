"""Tests for JITWriter."""

from absl.testing import parameterized
import chex
from jax import numpy as jnp

import multiscope
from multiscope import mjax
from absl.testing import absltest


def square(mjax_handle, x):
  """Function to calculate x^2 and log it to multiscope."""
  x = mjax.write_if_active(mjax_handle, 'X', x)
  x_square = x**2
  x_square = mjax.write_if_active(mjax_handle, 'X^2', x_square)
  return x_square


class JITWriterTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    multiscope.start_server()

  @chex.all_variants
  @parameterized.parameters(
      (3, 9),
      ([1, 2], [1, 4]),
  )
  def test(self, x, x_square):
    writer = mjax.JITWriter('test')
    result = self.variant(square)(writer.handle(), jnp.asarray(x))
    chex.assert_tree_all_close(result, jnp.asarray(x_square))


if __name__ == '__main__':
  absltest.main()
