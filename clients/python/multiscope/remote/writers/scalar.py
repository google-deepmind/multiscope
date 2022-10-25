"""Writes scalars to Multiscope."""
from typing import Any, Mapping, Optional

from multiscope.protos import scalar_pb2
from multiscope.protos import scalar_pb2_grpc
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base


class ScalarWriter(base.Writer):
  """Display a time series plot on the Multiscope page."""

  @control.init
  def __init__(self, name: str, parent: Optional[group.ParentNode] = None):

    self._client = scalar_pb2_grpc.ScalarsStub(stream_client.channel)
    path = group.join_path_pb(parent, name)
    req = scalar_pb2.NewWriterRequest(path=path)
    self.writer = self._client.NewWriter(req).writer

    super().__init__(path=tuple(self.writer.path.path))
    active_paths.register_callback(self.path, self._set_should_write)

    # TODO(b/251324180): this did not originally call `self._set_display()`
    # in the constructor. Consider making it consistent.


  @control.method
  def write(self, values: Mapping[str, Any]):
    """Write data to the vega writer."""
    converted_values: dict[str, float] = {}
    for k, v in values.items():
      try:
        # TODO: add comment on why this is here.
        for sub_key, sub_val in enumerate(v):
          converted_values[k + str(sub_key)] = float(sub_val)
      except TypeError:
        converted_values[k] = float(v)

    if not converted_values:
      return

    request = scalar_pb2.WriteRequest(
        writer=self.writer,
        label_to_value=converted_values,
    )
    self._client.Write(request)
