"""Example to test Multiscope with JAX and the reflect package."""

from absl import flags
from jax import random

from absl import app
import multiscope
# We import jax Multiscope to register the JAX parser.
from multiscope import mjax  # pylint: disable=unused-import
from multiscope.examples import examples

FLAGS = flags.FLAGS


class Experiment:
  """Updates some data."""

  def __init__(self):
    self.rng = random.PRNGKey(0)
    self.x = random.normal(self.rng, (50, 50))
    self.step()

  def step(self):
    self.rng, rng_input = random.split(self.rng)
    self.delta_x = random.normal(rng_input, self.x.shape)
    self.x = self.x + self.delta_x

  @multiscope.reflect_attrs
  def export_attrs(self):
    return multiscope.all_non_callable_attrs(self)


def main(_):
  multiscope.start_server()
  ticker = multiscope.Ticker(name='main')
  experiment = Experiment()
  multiscope.reflect(ticker, 'Experiment', experiment)
  for _ in examples.step():
    experiment.step()
    ticker.tick()


if __name__ == '__main__':
  app.run(main)
