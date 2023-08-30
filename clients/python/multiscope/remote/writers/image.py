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
"""Image writer."""
import io
from typing import Optional, Text

from multiscope.protos import base_pb2
from multiscope.protos import base_pb2_grpc
from multiscope.remote.control import decorators
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base
import numpy as np
from PIL import Image


class ImageWriter(base.Writer):
  """ImageWriter encapsulates a RawWriter in Multiscope."""

  @decorators.init
  def __init__(
      self,
      py_client: stream_client.Client,
      name: Text,
      parent: Optional[group.ParentNode] = None,
  ):
    self._client = base_pb2_grpc.BaseWritersStub(py_client.Channel())
    path = group.join_path_pb(parent, name)
    req = base_pb2.NewRawWriterRequest(
        tree_id=py_client.TreeID(), path=path, mime='image/png')
    self.writer = self._client.NewRawWriter(req).writer

    super().__init__(py_client=py_client, path=tuple(self.writer.path.path))
    self.register_activity_callback(self._set_should_write)

  @decorators.method
  def write(self, image: np.ndarray):
    assert image.dtype == np.dtype('uint8')
    png_buffer = io.BytesIO()
    Image.fromarray(image).save(png_buffer, format='png', compress_level=1)

    req = base_pb2.WriteRawRequest(writer=self.writer)
    req.data = png_buffer.getvalue()
    self._client.WriteRaw(req)
