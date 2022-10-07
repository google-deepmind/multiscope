import threading

from golang.stream import stream_pb2 as pb
from multiscope import remote as multiscope
from multiscope.remote import active_paths
from multiscope.remote import stream_client
from absl.testing import absltest


class ActivePathTest(absltest.TestCase):

  def testActivePath(self):
    """Tests if active_paths calls callbacks."""
    multiscope.start_server()

    path_ab = ["a", "b"]
    is_active_ab = threading.Event()

    def active_ab(_: bool):
      is_active_ab.set()

    active_paths.register_callback(tuple(path_ab), active_ab)

    path_a = ["a"]
    is_active_a = threading.Event()

    def active_a(_: bool):
      is_active_a.set()

    active_paths.register_callback(tuple(path_a), active_a)

    stream_client.GetNodeData(
        pb.NodeDataRequest(paths=[
            pb.NodePath(path=path_ab),
        ]))
    is_active_ab.wait()
    self.assertFalse(is_active_a.is_set())

    is_active_a.clear()
    is_active_ab.clear()
    stream_client.GetNodeData(
        pb.NodeDataRequest(paths=[
            pb.NodePath(path=path_a),
        ]))
    is_active_a.wait()
    is_active_ab.wait()


if __name__ == "__main__":
  absltest.main()