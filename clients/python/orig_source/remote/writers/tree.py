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

"""A tree writer."""
from typing import Optional



from golang.multiscope.streams import writers_pb2 as writers_pb
from golang.stream import stream_pb2 as pb
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base


class TreeWriter(base.Writer):
  """Writes a tree (in the d3-hierarchy json format) to multiscope."""

  @control.init
  def __init__(self, name: str, parent: Optional[group.ParentNode] = None):

    path = group.join_path(parent, name)
    request = pb.CreateNodeRequest()
    request.path.path.extend(path)
    request.type = writers_pb.Writers().raw_writer
    # Based on
    # https://json-schema.org/draft/2019-09/json-schema-core.html#rfc.section.13.2
    # but without a URI for the d3-hierarchy schema.
    request.mime = 'application/d3-hierarchy-instance+json'
    resp = stream_client.CreateNode(request=request)
    super().__init__(path=tuple(resp.path.path))
    active_paths.register_callback(self.path, self._set_should_write)
    self._set_display()


  @control.method
  def write(self, json: str):
    """Write a tree in json format.

    Args:
      json: that can be passed into d3.hierarchy.
    See https://github.com/d3/d3-hierarchy/blob/master/README.md#hierarchy
    """
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.path)
    request.data.raw = json.encode('utf-8')
    stream_client.PutNodeData(request=request)
