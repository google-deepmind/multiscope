# Copyright 2023 DeepMind Technologies Limited
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

from multiscope.protos import tree_pb2 as pb
from multiscope.remote import active_paths
from multiscope.remote import server as multiscope
from multiscope.remote import stream_client


class ActivePathTest(absltest.TestCase):

  def testActivePath(self):
    """Tests if active_paths calls callbacks."""
    multiscope.start_server(port=0)

    path_ab = ["a", "b"]
    is_active_ab = threading.Event()

    def active_ab(_: bool):
      is_active_ab.set()

    stream_client.GlobalClient().ActivePaths().register_callback(
        tuple(path_ab), active_ab)

    path_a = ["a"]
    is_active_a = threading.Event()

    def active_a(_: bool):
      is_active_a.set()

    stream_client.GlobalClient().ActivePaths().register_callback(
        tuple(path_a), active_a)
    tree_id = stream_client.GlobalClient().TreeID()
    stream_client.GetNodeData(
        pb.NodeDataRequest(
            tree_id=tree_id,
            reqs=[
                pb.DataRequest(path=pb.NodePath(path=path_ab), lastTick=0),
            ]))
    is_active_ab.wait()
    self.assertFalse(is_active_a.is_set())

    is_active_a.clear()
    is_active_ab.clear()
    stream_client.GetNodeData(
        pb.NodeDataRequest(
            tree_id=tree_id,
            reqs=[
                pb.DataRequest(path=pb.NodePath(path=path_a), lastTick=0),
            ]))
    is_active_a.wait()
    is_active_ab.wait()


if __name__ == "__main__":
  absltest.main()
