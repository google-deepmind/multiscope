"""The vega module is the stub to the Go Vega writer."""
import json
from typing import Any, Callable, Mapping, Optional, Text

import altair as alt

from golang.multiscope.streams import writers_pb2 as writers_pb
from golang.multiscope.streams.scalars import scalar_pb2 as scalar_pb
from golang.stream import stream_pb2 as pb
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base

import six


class ScalarWriter(base.Writer):
  """Display a time series plot on the Multiscope page."""

  @control.init
  def __init__(self,
               name: Text,
               parent: Optional[group.ParentNode] = None,
               history_length: Optional[int] = None,
               spec=None):

    request = pb.CreateNodeRequest()
    request.path.path.extend(group.join_path(parent, name))
    request.type = writers_pb.Writers().scalar_writer
    resp = stream_client.CreateNode(request=request)

    super().__init__(path=tuple(resp.path.path))
    self.spec_path = self.path + ('specification',)
    self.data_path = self.path + ('data',)
    self.register_activity_callback(self._set_should_write)

    self.chart = None
    self._last_chart = self.chart

    if history_length:
      self.set_history_length(history_length)
    if spec:
      self.set_spec(spec)

  @control.method
  def reset(self):
    """Reset the writer data."""
    action = scalar_pb.ScalarAction()
    action.reset.SetInParent()
    self._send_action(action)

  @control.method
  def write(self, values: Mapping[Text, Any]):
    """Write data to the vega writer."""
    if self._last_chart is not self.chart:
      if self.chart is not None:
        self.set_spec(self.chart)
      self._last_chart = self.chart

    converted_values = {}
    for k, v in values.items():
      try:
        for sub_key, sub_val in enumerate(v):
          converted_values[k + str(sub_key)] = float(sub_val)
      except TypeError:
        converted_values[k] = float(v)

    if not converted_values:
      return

    action = scalar_pb.ScalarAction()
    action.data.data.update(converted_values)
    self._send_action(action)

  def register_activity_callback(self, cb: Callable[[bool], None]):
    active_paths.register_callback(self.data_path, cb)  # pytype: disable=wrong-arg-types

  @property
  def altair_data(self):
    return alt.Data(values=[])

  @control.method
  def set_history_length(self, history_length):
    update = scalar_pb.ScalarAction()
    update.set_history_length.length = history_length
    self._send_action(update)

  @control.method
  def set_spec(self, spec: Any):
    if isinstance(spec, alt.Chart):
      spec = spec.to_json()
    elif not isinstance(spec, six.string_types):
      spec = json.dumps(spec)
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.spec_path)
    request.data.raw = spec.encode('utf-8')
    stream_client.PutNodeData(request=request)

  def _send_action(self, action: scalar_pb.ScalarAction, **kwargs):
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.path)
    request.data.pb.Pack(action)
    resp = stream_client.PutNodeData(request=request, **kwargs)
    return resp
