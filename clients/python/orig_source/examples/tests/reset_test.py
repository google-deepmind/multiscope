"""Tests for multiscope.examples."""

from absl import flags
from absl.testing import flagsaver

from multiscope.examples import reset
from absl.testing import absltest

FLAGS = flags.FLAGS

NSTEPS = 100


class ExampleTest(absltest.TestCase):

  @flagsaver.flagsaver
  def test_example(self):
    FLAGS.step_limit = NSTEPS
    reset.main(None)


if __name__ == '__main__':
  absltest.main()