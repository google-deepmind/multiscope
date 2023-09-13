"""Functions related to manage events coming from the GUI." """

from typing import Callable

from multiscope.protos import events_pb2
from multiscope.protos import root_pb2
from multiscope.protos import root_pb2_grpc
from multiscope.protos import tree_pb2
from multiscope.remote import stream_client
from multiscope.remote.events import events

mouse_event_uri = events_pb2.Mouse().DESCRIPTOR.full_name
keyboard_event_uri = events_pb2.Keyboard().DESCRIPTOR.full_name


def _enable_capture(py_client: stream_client.Client):
  root = root_pb2_grpc.RootStub(py_client.Channel())
  req = root_pb2.SetCaptureRequest(tree_id=py_client.TreeID(), capture=True)
  root.SetCapture(req)


def register_mouse_callback(
    py_client: stream_client.Client,
    cb: Callable[[events_pb2.Mouse], None],
):
  """Calls the provided cb with every mouse event at the provided path in a separate thread."""
  _enable_capture(py_client)

  def process(ev: tree_pb2.Event):
    mev = events_pb2.Mouse()
    mev.ParseFromString(ev.payload.value)
    cb(mev)

  process = events.filter_proto(mouse_event_uri, process)
  py_client.Events().register_callback([], process)


def register_keyboard_callback(
    py_client: stream_client.Client,
    cb: Callable[[events_pb2.Keyboard], None],
):
  """Calls the provided cb with every keyboard event at the provided path in a separate thread."""
  _enable_capture(py_client)

  def process(ev: tree_pb2.Event):
    kev = events_pb2.Keyboard()
    kev.ParseFromString(ev.payload.value)
    cb(kev)

  process = events.filter_proto(keyboard_event_uri, process)
  py_client.Events().register_callback([], process)
