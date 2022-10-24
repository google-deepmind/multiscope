import math

from absl.testing import absltest

from multiscope import remote
from multiscope.remote.writers import scalar


def setUpModule():
  # TODO: this had a timeout of 0 before. Is that because we don't actually need
  # to connect to multiscope for these tests? Consider best way forward.
  remote.start_server()


# TODO: use unittest.mock (from unittest import mock) and
#     mock_client = mock.create_autospec(
#         scalar_pb2_grpc.ScalarsStub, instance=True)
# instead of hand-defining the class here. Also consider patching it!
# Had an issue when I tried:
# `AttributeError: Mock object has no attribute 'Write'`
# -- maybe it has to do with it being a proto? Or I'm making a mistake.

class MockClient:

  def Write(self, request):
    self.request = request


class TestScopeScalarWriter(absltest.TestCase):

  def test_writer_dict(self):
    """Can write to a ScalarWriter."""
    w = scalar.ScalarWriter("sincos")
    for t in range(0, 10):
      w.write({
          "sin": math.sin(t * 0.01),
          "cos": math.cos(t * 0.05),
      })

  def test_writer_data_dict_vals(self):
    """Checks the outcome of the writing."""
    w = scalar.ScalarWriter("sincos")
    w._client = MockClient()
    w.write({
        "sin": math.sin(0.),
        "cos": math.cos(0.),
    })
    self.assertEqual(w._client.request.label_to_value, {"sin": 0., "cos": 1.})

    # TODO: when using a mock, do this:
    # mock_client.Write.assert_called_once()
    # _, write_kwargs = mock_client.call_args
    # self.assertEqual(write_kwargs["label_to_value"], {"sin": 0., "cos": 1.})

  def test_writer_sequence_vals(self):
    """Writes a sequence of values inside a dictionary."""
    w = scalar.ScalarWriter("sincos")
    for t in range(0, 10):
      w.write({"both": (math.sin(t * 0.01), math.cos(t * 0.05))})

  def test_writer_data_sequence_vals(self):
    """Checks the outcome of writing a sequence of values inside a dict."""
    w = scalar.ScalarWriter("sincos")
    w._client = MockClient()
    w.write({"both": (math.sin(0.), math.cos(0.))})
    self.assertEqual(
      w._client.request.label_to_value, {"both0": 0., "both1": 1.})

  def test_writer_nan(self):
    """Write NaN and Inf."""
    w = scalar.ScalarWriter("nan")
    for _ in range(0, 10):
      w.write({
          "nan": float("NaN"),
          "inf": float("Inf"),
      })


if __name__ == "__main__":
  absltest.main()
