# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from absl.testing import absltest

import multiscope

from multiscope.protos import tree_pb2
from multiscope.remote import stream_client
from multiscope.remote.writers import tensor


def setUpModule():
    # TODO: in some tests `start_server()` is imported from `remote.__init__`,
    #   here it's from multiscope. Make consistent.
    multiscope.start_server()


class TestTensorWriter(absltest.TestCase):
    def testWriter(self):
        """Creates a Tensor writer and then writes to it."""
        w = tensor.TensorWriter("writer")
        tensor_data = np.arange(0, 5, 1, dtype=np.int16)
        pb_path = tree_pb2.NodePath(path=w.path + ("tensor",))
        data_req = tree_pb2.DataRequest(path=pb_path, lastTick=0)
        node_data_req = tree_pb2.NodeDataRequest()
        node_data_req.reqs.append(data_req)
        empty_data = True
        while empty_data:
            w.write(tensor_data)
            pb_data = stream_client.GetNodeData(node_data_req)
            self.assertEmpty(pb_data.node_data[0].error)
            empty_data = not bool(pb_data.node_data[0].raw)
        self.assertNotEmpty(pb_data.node_data[0].raw)

    def testImgWriter(self):
        """Writes a tensor that is meant to be an image."""
        w = tensor.TensorWriter("img_writer")
        shape = (12, 7, 3)
        total_size = np.prod(shape)
        assert total_size < np.iinfo(np.uint8).max
        tensor_data = np.arange(0, total_size, 1, dtype=np.uint8)
        w.write(tensor_data)


if __name__ == "__main__":
    absltest.main()
