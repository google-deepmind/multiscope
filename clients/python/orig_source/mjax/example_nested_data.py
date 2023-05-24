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

"""Example to test Multiscope with JAX tensors in nested data structures."""

import collections

from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from absl import app
import multiscope
from multiscope import mjax
from multiscope.examples import examples

FLAGS = flags.FLAGS

TwoThings = collections.namedtuple('TwoThings', 'first second')


def check_haiku():
  """Check if haiku data types work with NestedWriter."""
  multiscope.start_server()

  data_x = jnp.array(np.random.random((20, 10, 10)))
  data_y = jnp.array(np.random.random((20, 10)))

  def raw(x):
    f = hk.Sequential([
        hk.Flatten(),
        hk.Linear(10),
        jax.nn.relu,
        hk.Linear(10),
    ])
    return f(x)

  net = hk.without_apply_rng(hk.transform(raw))
  ticker = examples.Ticker('main')
  agent = mjax.NestedWriter('agent')

  opt = optax.adam(1e-3)
  params = net.init(jax.random.PRNGKey(42), data_x)
  opt_state = opt.init(params)

  def loss(params, x, y):
    yp = net.apply(params, x)
    return 0.5 * jnp.sum(jnp.square(yp - y))

  for _ in examples.step():
    grads = jax.grad(loss)(params, data_x, data_y)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    agent.write({'weights': params, 'optimizer_state': opt_state})
    ticker.tick()


def check_container(container_reader):
  """Check if the given container type works with NestedWriter."""
  multiscope.start_server()

  container, reader = container_reader()
  data = container(jnp.ones((10, 15)), jnp.zeros((10, 15)))
  common = 0.05 * jnp.array(np.random.random((10, 15)))

  ticker = examples.Ticker('main')
  agent = mjax.NestedWriter('agent')
  for _ in examples.step():
    p = reader(data) + common
    q = jnp.sin(p)
    data = container(p, q)
    agent.write(data)
    ticker.tick()


def check_wrong_content():
  """Check if the given container type works with NestedWriter."""
  multiscope.start_server()

  data = tuple([jnp.ones((10, 15)), jnp.zeros((10, 15)),
                np.zeros(5), 'wrong content text', np.array(3), lambda x: x+4])
  common = 0.05 * jnp.array(np.random.random((10, 15)))

  ticker = examples.Ticker('main')
  agent = mjax.NestedWriter('agent')
  for _ in examples.step():
    p = data[0] + common
    q = jnp.sin(p)
    data = tuple([p, q, data[2], data[3], data[4], data[5]])
    agent.write(data)
    ticker.tick()


def tuple_funcs():
  def container(x, y):
    return (x, y)
  def reader(x):
    return x[0]
  return container, reader


def list_funcs():
  def container(x, y):
    return [x, y]
  def reader(x):
    return x[0]
  return container, reader


def named_tuple_funcs():
  def container(x, y):
    return TwoThings(first=x, second=y)
  def reader(x):
    return x.first
  return container, reader


def dict_funcs():
  def container(x, y):
    return {'first': x, 'second': y}
  def reader(x):
    return x['first']
  return container, reader


def nested_funcs():
  def container(x, y):
    return [(x, y), y]
  def reader(x):
    return x[0][0]
  return container, reader


def main(_):
  check_haiku()


if __name__ == '__main__':
  app.run(main)
