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
"""A text writer."""
from typing import Optional

from multiscope.protos import text_pb2
from multiscope.protos import text_pb2_grpc
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base


class TextWriter(base.Writer):
  """Writes text on the Multiscope page."""

  @control.init
  def __init__(self, name: str, parent: Optional[group.ParentNode] = None):
    self._client = text_pb2_grpc.TextStub(stream_client.channel)
    path = group.join_path_pb(parent, name)
    req = text_pb2.NewWriterRequest(path=path)
    self.writer = self._client.NewWriter(req).writer
    super().__init__(path=tuple(self.writer.path.path))
    active_paths.register_callback(self.path, self._set_should_write)

    # TODO(b/251324180): re-enable once fixed.
    # self._set_display()

  @control.method
  def write(self, data: str):
    request = text_pb2.WriteRequest(
        writer=self.writer,
        # type-error suggest that we should pass `str` not `bytes` here,
        # ignoring for now.
        text=data.encode("utf-8"),  # pytype: disable=wrong-arg-types
    )
    self._client.Write(request)
