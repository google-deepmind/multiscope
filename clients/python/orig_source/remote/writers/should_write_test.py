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
