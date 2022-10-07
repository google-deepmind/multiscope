"""Tests for the example in the jax package."""

from absl import flags
from absl.testing import flagsaver

from multiscope.mjax import example_conditional_writers
from multiscope.mjax import example_nested_data
from multiscope.mjax import example_reflect
from absl.testing import absltest

FLAGS = flags.FLAGS

NSTEPS = 100


class ExampleTest(absltest.TestCase):

  @flagsaver.flagsaver
  def test_example_reflect(self):
    FLAGS.step_limit = NSTEPS
    example_reflect.main(None)

  @flagsaver.flagsaver
  def test_example_lazy_writers(self):
    FLAGS.step_limit = NSTEPS
    example_conditional_writers.main(None)

  @flagsaver.flagsaver
  def test_example_nested_data_tuples(self):
    FLAGS.step_limit = NSTEPS
    example_nested_data.check_container(example_nested_data.tuple_funcs)

  @flagsaver.flagsaver
  def test_example_nested_data_list(self):
    FLAGS.step_limit = NSTEPS
    example_nested_data.check_container(example_nested_data.list_funcs)

  @flagsaver.flagsaver
  def test_example_nested_data_named_tuple(self):
    FLAGS.step_limit = NSTEPS
    example_nested_data.check_container(example_nested_data.named_tuple_funcs)

  @flagsaver.flagsaver
  def test_example_nested_data_dict(self):
    FLAGS.step_limit = NSTEPS
    example_nested_data.check_container(example_nested_data.dict_funcs)

  @flagsaver.flagsaver
  def test_example_nested_data_nested(self):
    FLAGS.step_limit = NSTEPS
    example_nested_data.check_container(example_nested_data.nested_funcs)

  @flagsaver.flagsaver
  def test_example_nested_data_haiku(self):
    FLAGS.step_limit = NSTEPS
    example_nested_data.check_haiku()


if __name__ == '__main__':
  absltest.main()