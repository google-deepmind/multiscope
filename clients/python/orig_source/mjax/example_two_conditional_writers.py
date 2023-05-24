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

"""Example of linear regression with multiple JITWriters."""

from absl import flags
import jax
import jax.numpy as jnp

from absl import app
import multiscope
from multiscope import mjax
from multiscope.examples import examples

FLAGS = flags.FLAGS


def infer(params, xs, mjax_handle):
  xs = mjax.write_if_active(mjax_handle, 'input_xs', xs)
  ys = jnp.matmul(params, xs)
  ys = mjax.write_if_active(mjax_handle, 'ys', ys)
  return ys


def loss(params, xs, ts, mjax_handle):
  ys = infer(params, xs, mjax_handle)
  l = jnp.mean((ys - ts)**2)
  l = mjax.write_if_active(mjax_handle, 'loss', jnp.array([l]))
  return l[0]


def update(params, xs, ts, mjax_handle):
  """JIT function to update params."""
  d_params = jax.grad(loss)(params, xs, ts, mjax_handle)
  d_params = mjax.write_if_active(mjax_handle, 'd_params', d_params)

  params = params - 0.1 * d_params
  params = mjax.write_if_active(mjax_handle, 'new_params', params)
  return params


def main(_):
  multiscope.start_server()

  # Generate data.
  rng, rng_p, rng_ts, rng_coeff = jax.random.split(jax.random.PRNGKey(1), 4)
  params = jax.random.normal(rng_p, (1, 50))
  xs_test = jax.random.normal(rng_ts, (50, 50))
  coeff = 100 * jax.random.normal(rng_coeff, (1, 50))
  ts_test = jnp.matmul(coeff, xs_test)

  # Compile methods.
  update_jit = jax.jit(update)
  loss_jit = jax.jit(loss)

  ticker = examples.Ticker('main')
  writer_train = mjax.JITWriter('train')
  writer_test = mjax.JITWriter('test')

  # Training loop.
  while ticker.tick():
    rng, rng_tr, rng_noise = jax.random.split(rng, 3)
    xs_train = jax.random.normal(rng_tr, (50, 50))
    ts_train = jnp.matmul(
        coeff, xs_train) + 1e3 * jax.random.normal(rng_noise, (1, 50))

    _ = loss_jit(params, xs_test, ts_test, writer_test.handle())
    params = update_jit(params, xs_train, ts_train, writer_train.handle())


if __name__ == '__main__':
  app.run(main)
