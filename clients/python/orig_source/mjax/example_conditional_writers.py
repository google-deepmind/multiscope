"""Example to test Multiscope with JAX."""

from absl import flags
import jax

from absl import app
import multiscope
from multiscope import mjax
from multiscope.examples import examples

FLAGS = flags.FLAGS


def update(mjax_handle, rng_input, x):
  """JIT function to update x."""
  delta_x = jax.random.normal(rng_input, x.shape)
  # You need to store the result of write_if_active into the monitored tensor,
  # otherwise the compiler will remove the call.
  delta_x = mjax.write_if_active(mjax_handle, 'Delta X', delta_x)
  step_size = 0.001
  x = (1 - step_size) * x + step_size * delta_x
  # You need to store the result of write_if_active into the monitored tensor,
  # otherwise the compiler will remove the call.
  x = mjax.write_if_active(mjax_handle, 'X', x)
  return x


def main(_):
  multiscope.start_server()

  rng = jax.random.PRNGKey(0)
  x = jax.random.normal(rng, (50, 50))
  update_jit = jax.jit(update)
  ticker = examples.Ticker('main')
  writer = mjax.JITWriter('update')
  while ticker.tick():
    rng, rng_input = jax.random.split(rng)
    x = update_jit(writer.handle(), rng_input, x)


if __name__ == '__main__':
  app.run(main)
