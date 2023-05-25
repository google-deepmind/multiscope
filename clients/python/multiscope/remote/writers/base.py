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
"""Base writer class."""

import abc
import threading
from typing import Callable, Tuple

# from multiscope import root_pb2 as root_pb
# from multiscope.protos import tree_pb2 as stream_pb2
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import stream_client


class Node(abc.ABC):

  def __init__(self, path: Tuple[str, ...]):
    self.path = path
    self.name = self.path[-1]


class Writer(Node):
  """A writer has a write method.."""

  def __init__(self, path: Tuple[str, ...]):
    Node.__init__(self, path=path)
    self._should_write = False
    self._should_write_lock = threading.Lock()
    # This property is controlled by the decorators in
    # l/d/v/multiscope/remote/control/decorators.py.
    self.enabled = True

  @property
  def should_write(self) -> bool:
    """Returns True when the writer is active.

        For example, a user is currently watching the data being written.
        """
    if not self.enabled:
      return False
    with self._should_write_lock:
      return self._should_write

  def _set_should_write(self, is_active: bool):
    with self._should_write_lock:
      self._should_write = is_active

  # TODO(b/251324180): re-enable this.
  # def _set_display(self) -> stream_pb2.PutNodeDataReply:
  #   action = root_pb.RootAction()
  #   node_path = stream_pb2.NodePath()
  #   node_path.path.extend(self.path)
  #   action.display.add_display_paths.append(node_path)
  #   request = stream_pb2.PutNodeDataRequest()
  #   # Sets an empty path corresponding to multiscope root.
  #   request.data.path.SetInParent()
  #   request.data.pb.Pack(action)
  #   return stream_client.PutNodeData(request=request)

  def register_activity_callback(self, cb: Callable[[bool], None]):
    active_paths.register_callback(self.path, cb)

  def reset(self):
    """Clears the data of the writer in the backend, if any.

        For example: clears history for the scalar writer.
        When using reflection, the writer's data is cleared when it becomes inactive
        to avoid confusion due to stale data when reopening the dashboard.
        """

  @abc.abstractmethod
  def write(self, *args, **kwargs):
    """Writes to Multiscope."""

  def flush(self):
    pass

  def close(self):
    pass
