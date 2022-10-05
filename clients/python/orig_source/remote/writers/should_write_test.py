
import threading

from absl.testing import absltest

from golang.stream import stream_pb2 as pb
import multiscope
from multiscope.remote import active_paths
from multiscope.remote import stream_client


def setUpModule():
  multiscope.start_server()


class ShouldWriteTest(absltest.TestCase):

  def testScalarWriter(self):
    node_name = "testScalarWriter"
    w = multiscope.ScalarWriter(node_name)
    self.assertFalse(w.should_write)

    is_active_w = threading.Event()

    def active_w(_: bool):
      is_active_w.set()

    active_paths.register_callback(w.data_path, active_w)  # pytype: disable=wrong-arg-types

    stream_client.GetNodeData(
        pb.NodeDataRequest(paths=[
            pb.NodePath(path=w.data_path),
        ]))
    is_active_w.wait()
    self.assertTrue(w.should_write)


if __name__ == "__main__":
  absltest.main()
