import math

from absl.testing import absltest

from multiscope import remote
from multiscope.remote.writers import scalar


def setUpModule():
    remote.start_server(connection_timeout_secs=2)


class MockClient:
    def Write(self, request):
        self.request = request


class TestScopeScalarWriter(absltest.TestCase):
    def test_writer_dict(self):
        """Can write to a ScalarWriter."""
        w = scalar.ScalarWriter("sincos")
        for t in range(0, 10):
            w.write(
                {
                    "sin": math.sin(t * 0.01),
                    "cos": math.cos(t * 0.05),
                }
            )

    def test_writer_data_dict_vals(self):
        """Checks the outcome of the writing."""
        w = scalar.ScalarWriter("sincos")
        w._client = MockClient()
        w.write(
            {
                "sin": math.sin(0.0),
                "cos": math.cos(0.0),
            }
        )
        self.assertEqual(w._client.request.label_to_value, {"sin": 0.0, "cos": 1.0})

    def test_writer_sequence_vals(self):
        """Writes a sequence of values inside a dictionary."""
        w = scalar.ScalarWriter("sincos")
        for t in range(0, 10):
            w.write({"both": (math.sin(t * 0.01), math.cos(t * 0.05))})

    def test_writer_data_sequence_vals(self):
        """Checks the outcome of writing a sequence of values inside a dict."""
        w = scalar.ScalarWriter("sincos")
        w._client = MockClient()
        w.write({"both": (math.sin(0.0), math.cos(0.0))})
        self.assertEqual(w._client.request.label_to_value, {"both0": 0.0, "both1": 1.0})

    def test_writer_nan(self):
        """Writes NaN and Inf."""
        w = scalar.ScalarWriter("nan")
        for _ in range(0, 10):
            w.write(
                {
                    "nan": float("NaN"),
                    "inf": float("Inf"),
                }
            )


if __name__ == "__main__":
    absltest.main()
