"""Tests for multiscope.examples."""

from absl import flags
from absl.testing import flagsaver

from multiscope.examples import echo
from absl.testing import absltest

FLAGS = flags.FLAGS

NSTEPS = 5


class ExampleTest(absltest.TestCase):

  @flagsaver.flagsaver
  def test_example(self):
    FLAGS.step_limit = NSTEPS
    echo.main(None)


if __name__ == '__main__':
  absltest.main()
