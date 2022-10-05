"""Tests for multiscope.examples."""

from absl import flags
from absl.testing import flagsaver

from multiscope.examples import clock
from absl.testing import absltest

FLAGS = flags.FLAGS

NSTEPS = 20


class ExampleTest(absltest.TestCase):

  @flagsaver.flagsaver
  def test_example(self):
    FLAGS.step_limit = NSTEPS
    clock.main(None)


if __name__ == '__main__':
  absltest.main()
