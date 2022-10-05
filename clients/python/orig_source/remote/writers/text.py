"""A text writer."""
from typing import Optional, Text

from golang.multiscope.streams import writers_pb2 as writers_pb
from golang.stream import stream_pb2 as pb
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base


# TODO: Write some tests for this.  # pylint: disable=g-bad-todo
class TextWriter(base.Writer):
  """Writes text on the Multiscope page."""

  @control.init
  def __init__(self, name: Text, parent: Optional[group.ParentNode] = None):
    path = group.join_path(parent, name)
    request = pb.CreateNodeRequest()
    request.path.path.extend(path)
    request.type = writers_pb.Writers().text_writer
    resp = stream_client.CreateNode(request=request)
    super().__init__(path=tuple(resp.path.path))
    active_paths.register_callback(self.path, self._set_should_write)

  @control.method
  def write(self, data: Text):
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.path)
    request.data.raw = data.encode('utf-8')
    stream_client.PutNodeData(request=request)
