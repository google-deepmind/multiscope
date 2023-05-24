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

from typing import Optional, Text



from golang.multiscope.streams import writers_pb2 as writers_pb
from golang.stream import stream_pb2 as pb
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base


class HTMLWriter(base.Writer):
  """Writes text on the Multiscope page."""

  @control.init
  def __init__(self, name: Text, parent: Optional[group.ParentNode] = None):

    path = group.join_path(parent, name)
    request = pb.CreateNodeRequest()
    request.path.path.extend(path)
    request.type = writers_pb.Writers().html_writer
    resp = stream_client.CreateNode(request=request)
    super().__init__(path=tuple(resp.path.path))
    self.html_path = self.path + ('html',)
    self.css_path = self.path + ('css',)
    active_paths.register_callback(self.html_path, self._set_should_write)  # pytype: disable=wrong-arg-types


  @control.method
  def write(self, html: Text):
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.html_path)
    request.data.raw = html.encode('utf-8')
    stream_client.PutNodeData(request=request)


  @control.method
  def writeCSS(self, css: Text):
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.css_path)
    request.data.raw = css.encode('utf-8')
    stream_client.PutNodeData(request=request)
