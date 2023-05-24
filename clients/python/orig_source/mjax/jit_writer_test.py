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
