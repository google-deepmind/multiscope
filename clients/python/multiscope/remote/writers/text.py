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
"""A text writer."""
from typing import Optional

from multiscope.protos import text_pb2
from multiscope.protos import text_pb2_grpc
from multiscope.remote import active_paths
from multiscope.remote.control import control
from multiscope.remote.control import decorators
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base


class TextWriter(base.Writer):
  """Writes text on the Multiscope page."""

  @decorators.init
  def __init__(
      self,
      py_client: stream_client.Client,
      name: str,
      parent: Optional[group.ParentNode] = None,
  ):
    self._client = text_pb2_grpc.TextStub(self._py_client.Channel())
    path = group.join_path_pb(parent, name)
    req = text_pb2.NewWriterRequest(tree_id=self._py_client.TreeID(), path=path)
    self.writer = self._client.NewWriter(req).writer
    super().__init__(py_client=py_client, path=tuple(self.writer.path.path))
    self._py_client.ActivePaths().register_callback(self.path,
                                                    self._set_should_write)

  @decorators.method
  def write(self, data: str):
    request = text_pb2.WriteRequest(
        writer=self.writer,
        text=data.encode("utf-8"),  # pytype: disable=wrong-arg-types
    )
    self._client.Write(request)


class HTMLWriter(base.Writer):
  """Writes HTML on the Multiscope page."""

  @control.init
  def __init__(
      self,
      py_client: stream_client.Client,
      name: str,
      parent: Optional[group.ParentNode] = None,
  ):
    self._client = text_pb2_grpc.TextStub(py_client.Channel())
    path = group.join_path_pb(parent, name)
    req = text_pb2.NewHTMLWriterRequest(
        tree_id=self._py_client.TreeID(), path=path)
    self.writer = self._client.NewHTMLWriter(req).writer
    super().__init__(py_client=py_client, path=tuple(self.writer.path.path))
    self._py_client.ActivePaths().register_callback(self.path,
                                                    self._set_should_write)

  @decorators.method
  def write_css(self, data: str):
    request = text_pb2.WriteCSSRequest(
        writer=self.writer,
        css=data.encode("utf-8"),  # pytype: disable=wrong-arg-types
    )
    self._client.WriteCSS(request)

  @decorators.method
  def write(self, data: str):
    request = text_pb2.WriteHTMLRequest(
        writer=self.writer,
        html=data.encode("utf-8"),  # pytype: disable=wrong-arg-types
    )
    self._client.WriteHTML(request)
