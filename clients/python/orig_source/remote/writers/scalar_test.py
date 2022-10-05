import math

from multiscope import remote
from multiscope.remote.writers import scalar
from absl.testing import absltest


def setUpModule():
  remote.start_server(0)


class SendActionMock:

  def __init__(self):
    self.data = None

  def __call__(self, scalar_action_pb):
    self.data = scalar_action_pb.data.data


class TestScopeScalarWriter(absltest.TestCase):

  def test_writer_dict(self):
    """Create a Vega Multiscope writer and then writes to it."""
    w = scalar.ScalarWriter("sincos")
    w.set_history_length(400)
    w.set_spec({})
    for t in range(0, 10):
      w.write({
          "sin": math.sin(t * 0.01),
          "cos": math.cos(t * 0.05),
      })

  def test_writer_data_dict_vals(self):
    """Create a Vega Multiscope writer and then writes to it."""
    send_action_mock = SendActionMock()
    w = scalar.ScalarWriter("sincos")
    w._send_action = send_action_mock
    w.set_history_length(400)
    w.set_spec({})
    w.write({
        "sin": math.sin(0.),
        "cos": math.cos(0.),
    })
    self.assertEqual(send_action_mock.data, {"sin": 0., "cos": 1.})

  def test_writer_sequence_vals(self):
    """Create a Vega Multiscope writer and then writes to it."""
    w = scalar.ScalarWriter("sincos")
    w.set_history_length(400)
    w.set_spec({})
    for t in range(0, 10):
      w.write({"both": (math.sin(t * 0.01), math.cos(t * 0.05))})

  def test_writer_data_sequence_vals(self):
    """Create a Vega Multiscope writer and then writes to it."""
    send_action_mock = SendActionMock()
    w = scalar.ScalarWriter("sincos")
    w._send_action = send_action_mock
    w.set_history_length(400)
    w.set_spec({})
    w.write({"both": (math.sin(0.), math.cos(0.))})
    self.assertEqual(send_action_mock.data, {"both0": 0., "both1": 1.})

  def test_writer_nan(self):
    """Create a Vega Multiscope writer and then writes to it."""
    w = scalar.ScalarWriter("nan")
    for _ in range(0, 10):
      w.write({
          "nan": float("NaN"),
          "inf": float("Inf"),
      })


if __name__ == "__main__":
  absltest.main()
