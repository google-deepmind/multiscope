"""A Multiscope clock for synchronizing writes."""
from typing import Optional, Text

from absl import logging

from golang.multiscope.streams import writers_pb2 as writers_pb
from golang.multiscope.streams.ticker import ticker_pb2 as ticker_pb
from golang.stream import stream_pb2 as pb
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client


class Clock(group.ParentNode):
  """Multiscope clock."""

  @control.init
  def __init__(self, name: Text, parent: Optional[group.ParentNode] = None):
    path = group.join_path(parent, name)
    request = pb.CreateNodeRequest()
    request.path.path.extend(path)
    request.type = writers_pb.Writers().ticker
    resp = stream_client.CreateNode(request=request)
    super().__init__(path=tuple(resp.path.path))
    self.listeners = []

  @control.method
  def register_listener(self, listener):
    """Register a listener which will be called just before each tick."""
    self.listeners.append(listener)

  @control.method
  def tick(self) -> bool:
    for listener in self.listeners:
      listener()
    action = ticker_pb.TickerAction()
    action.tick.SetInParent()
    # never time out as the clock may be paused
    self._send_action(action, timeout=int(1e9))
    return True

  @property
  @control.method
  def period(self):
    """Get the period of the clock."""
    request = pb.NodeDataRequest()
    path = pb.NodePath()
    path.path.extend(self.path)
    request.paths.append(path)
    node_data = stream_client.GetNodeData(request=request).node_data[0]
    if node_data.error:
      logging.error('Failed to get period for %s: %s', self.name,
                    node_data.error)
      return None
    else:
      ticker: ticker_pb.Ticker = ticker_pb.Ticker.FromString(node_data.pb.value)
      return ticker.clock_period

  @period.setter
  @control.method
  def period(self, p_nanos: int):
    action = ticker_pb.TickerAction()
    action.set_period.period_ms = p_nanos // int(1e6)
    self._send_action(action)

  @control.method
  def pause(self):
    """Pauses the clock on the next tick."""
    action = ticker_pb.TickerAction()
    action.command = ticker_pb.TickerAction.Command.PAUSE
    self._send_action(action)

  @control.method
  def run(self):
    """Runs the clock if paused, otherwise is a no-op."""
    action = ticker_pb.TickerAction()
    action.command = ticker_pb.TickerAction.Command.RUN
    self._send_action(action)

  def _send_action(self, action: ticker_pb.TickerAction, **kwargs):
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.path)
    request.data.pb.Pack(action)
    resp = stream_client.PutNodeData(request=request, **kwargs)
    return resp


Ticker = Clock
