"""Tests for the example in the reflect package."""

from absl import flags
from absl.testing import flagsaver

from multiscope.reflect import example
from absl.testing import absltest

FLAGS = flags.FLAGS

NSTEPS = 100


class ExampleTest(absltest.TestCase):

  @flagsaver.flagsaver
  def test_example(self):
    FLAGS.step_limit = NSTEPS
    example.main(None)


if __name__ == '__main__':
  absltest.main()
