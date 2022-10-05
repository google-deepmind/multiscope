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
