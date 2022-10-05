import numpy as np

from golang.stream import stream_pb2 as stream_pb
from multiscope import remote as multiscope
from multiscope.remote import stream_client
from absl.testing import absltest


def setUpModule():
  multiscope.start_server(0)


class TestTensorWriter(absltest.TestCase):

  def testWriter(self):
    """Creates a Tensor writer and then writes to it."""
    w = multiscope.TensorWriter('writer')
    tensor_data = np.arange(0, 5, 1, dtype=np.int16)
    pb_path = stream_pb.NodePath(path=w.path + ('image',))
    req = stream_pb.NodeDataRequest(paths=[pb_path])
    empty_data = True
    while empty_data:
      w.write(tensor_data)
      pb_data = stream_client.GetNodeData(req)
      self.assertEmpty(pb_data.node_data[0].error)
      empty_data = not bool(pb_data.node_data[0].raw)
    self.assertNotEmpty(pb_data.node_data[0].raw)


if __name__ == '__main__':
  absltest.main()
