"""Test for the multiscope with launchpad example."""

from absl.testing import absltest
import launchpad as lp
from multiscope.examples import launchpad as launchpad_example


class LaunchTest(absltest.TestCase):

  def test_consumer_steps(self):
    """Runs the program and makes sure the consumer can run 10 steps."""
    program = launchpad_example.make_program(num_nodes=2)

    nodes = program.groups['current_time']
    # Disable automatic `run`.
    for n in nodes:
      n.disable_run()

    lp.launch(
        program,
        launch_type='test_mp',
        local_resources=dict(
            current_time=lp.PythonProcess(
                interpreter=None,  # TODO: find an interpreter.  # pylint: disable=g-bad-todo
                args=dict(multi_port=0, multiscope_strict_mode=True))),
        test_case=self)

    for n in nodes:
      node = n.create_handle().dereference()
      # Success criteria for this integration test defined as node being
      # able to take 10 steps.
      for _ in range(10):
        node.step()


if __name__ == '__main__':
  absltest.main()