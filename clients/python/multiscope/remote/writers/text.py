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
            text=data.encode("utf-8"),
        )
        self._client.Write(request)
