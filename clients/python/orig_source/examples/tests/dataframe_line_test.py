"""Tests for multiscope.examples."""

from absl import flags
from absl.testing import flagsaver

from multiscope.examples import dataframe_line
from absl.testing import absltest

FLAGS = flags.FLAGS

NSTEPS = 100


class ExampleTest(absltest.TestCase):

  @flagsaver.flagsaver
  def test_example(self):
    FLAGS.step_limit = NSTEPS
    dataframe_line.main(None)


if __name__ == '__main__':
  absltest.main()
