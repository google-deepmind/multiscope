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

"""Image writer."""
import io
from typing import Optional, Text

import numpy as np
from PIL import Image



from golang.multiscope.streams import writers_pb2 as writers_pb
from golang.stream import stream_pb2 as pb
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base


class ImageWriter(base.Writer):
  """ImageWriter encapsulates a RawWriter in Multiscope."""

  @control.init
  def __init__(self, name: Text, parent: Optional[group.ParentNode] = None):

    path = group.join_path(parent, name)
    request = pb.CreateNodeRequest()
    request.path.path.extend(path)
    request.type = writers_pb.Writers().raw_writer
    request.mime = 'image/png'
    create_writer_resp = stream_client.CreateNode(request=request)

    super().__init__(path=tuple(create_writer_resp.path.path))
    active_paths.register_callback(self.path, self._set_should_write)
    self._custom_size = False
    self._image_height = None
    self._image_width = None

    self._set_display()


  @control.method
  def write(self, image: np.ndarray):
    assert image.dtype == np.dtype('uint8')
    png_buffer = io.BytesIO()
    Image.fromarray(image).save(png_buffer, format='png', compress_level=1)

    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.path)
    request.data.raw = png_buffer.getvalue()
    stream_client.PutNodeData(request=request)

    # (height, width, [channels])
    height, width = image.shape[0], image.shape[1]
    if not self._custom_size and (height != self._image_height or
                                  width != self._image_width):
      self.set_display_size(height=height, width=width)
      self._image_height = height
      self._image_width = width

  @control.method
  def set_display_size(self, height: int, width: int):
    super().set_display_size(height=height, width=width)
    self._custom_size = True
