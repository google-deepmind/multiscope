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

"""Module to register events callback with Multiscope.

Event handling in Python comes with caveats.

By default CPython3 will run a thread for a maximum of 5ms before preempting it
to release the GIL. A CPU-bound thread can starve an IO-bound thread of the GIL,
see https://bugs.python.org/issue7946.

If your main loop is CPU-bound, to ensure quicker preemption set:

sys.setswitchinterval(0.001)
"""
import threading
import types
from typing import Callable, Generator, Generic, Optional, Sequence, Text, Tuple, Type, TypeVar

from absl import logging

from golang.multiscope.streams.userinputs import userinputs_pb2
from golang.stream import stream_pb2 as pb
from multiscope.proto import webcam_pb2 as webcam_pb
from multiscope.remote import stream_client

_mouse_filter = userinputs_pb2.MouseEvent.DESCRIPTOR.full_name
_keyboard_filter = userinputs_pb2.KeyboardEvent.DESCRIPTOR.full_name
_gamepad_filter = userinputs_pb2.GamepadEvent.DESCRIPTOR.full_name
_webcam_filter = webcam_pb.WebcamEvent.DESCRIPTOR.full_name

_Path = Sequence[Text]
_EventPayload = Tuple[_Path, bytes]
_MousePayload = Tuple[_Path, userinputs_pb2.MouseEvent]
_KeyboardPayload = Tuple[_Path, userinputs_pb2.KeyboardEvent]
_GamepadPayload = Tuple[_Path, userinputs_pb2.GamepadEvent]
_WebcamPayload = Tuple[_Path, webcam_pb.WebcamEvent]

T = TypeVar('T')


class EventSubscription(Generic[T], Generator[T, None, None]):
  """Generator of events for a given path.

  This class exists rather than a generator function because we want the event
  subscription to happen synchronously on initialization, rather than first time
  the generator is called.
  """

  def __init__(self, path: Sequence[Text], type_url_filter: Text,
               process: Callable[[_EventPayload], T]):
    self._process = process
    request = pb.StreamEventsRequest()
    request.path.path.extend(path)
    request.type_url = type_url_filter
    self._queue = stream_client.StreamEvents(request=request)

  def send(self, value: None) -> T:
    try:
      # Blocks forever until the next value is available.
      event = next(self._queue)
      path, payload = list(event.path.path), event.payload.value
      return self._process((path, payload))
    except:
      self._queue.cancel()
      raise

  def throw(self,
            typ: Type[BaseException],
            val: Optional[BaseException] = ...,
            tb: Optional[types.TracebackType] = ...) -> None:  # pytype: disable=annotation-type-mismatch
    raise StopIteration


def subscribe_mouse_events(
    path: Sequence[Text]) -> EventSubscription[_MousePayload]:
  """Returns a blocking generator of mouse events for the given path."""

  def process(payload: _EventPayload) -> _MousePayload:
    event = userinputs_pb2.MouseEvent()
    event.ParseFromString(payload[1])
    return payload[0], event

  return EventSubscription(
      path=path, type_url_filter=_mouse_filter, process=process)


def subscribe_keyboard_events(
    path: Sequence[Text]) -> EventSubscription[_KeyboardPayload]:
  """Returns a blocking generator of keyboard events for the given path."""

  def process(payload: _EventPayload) -> _KeyboardPayload:
    event = userinputs_pb2.KeyboardEvent()
    event.ParseFromString(payload[1])
    return payload[0], event

  return EventSubscription(
      path=path, type_url_filter=_keyboard_filter, process=process)


def subscribe_gamepad_events(
    path: Sequence[Text]) -> EventSubscription[_GamepadPayload]:
  """Returns a blocking generator of mouse events for the given path."""

  def process(payload: _EventPayload) -> _GamepadPayload:
    event = userinputs_pb2.GamepadEvent()
    event.ParseFromString(payload[1])
    return payload[0], event

  return EventSubscription(
      path=path, type_url_filter=_gamepad_filter, process=process)


def subscribe_webcam_events(path: _Path):
  """Returns a blocking generator of webcam events for the given path."""

  def process(payload: _EventPayload) -> _WebcamPayload:
    event = webcam_pb.WebcamEvent()
    event.ParseFromString(payload[1])
    return payload[0], event

  return EventSubscription(
      path=path, type_url_filter=_webcam_filter, process=process)


def register_callback(gen: Generator[Tuple[_Path, T], None, None],
                      cb: Callable[[_Path, T], None]) -> None:
  """Calls the provided cb with every element of gen in a separate thread."""

  def _handler():
    try:
      for path, event in gen:
        cb(path, event)
    except:
      logging.exception('exception raised while handling events')
      raise

  threading.Thread(target=_handler, daemon=True).start()


def register_mouse_callback(cb: Callable[[_Path, userinputs_pb2.MouseEvent],
                                         None],
                            path: Optional[_Path] = None):
  """Calls the provided cb with every mouse event at the provided path in a separate thread."""
  if path is None:
    path = []
  register_callback(subscribe_mouse_events(path), cb)


def register_keyboard_callback(
    cb: Callable[[_Path, userinputs_pb2.KeyboardEvent], None],
    path: Optional[_Path] = None):
  """Calls the provided cb with every keyboard event at the provided path in a separate thread."""
  if path is None:
    path = []
  register_callback(subscribe_keyboard_events(path), cb)


def register_gamepad_callback(cb: Callable[[_Path, userinputs_pb2.GamepadEvent],
                                           None],
                              path: Optional[_Path] = None):
  """Calls the provided cb with every gamepad event at the provided path in a separate thread."""
  if path is None:
    path = []
  register_callback(subscribe_gamepad_events(path), cb)


def register_webcam_callback(cb: Callable[[_Path, webcam_pb.WebcamEvent], None],
                             path: _Path):
  """Calls the provided cb with every webcam event at the provided path in a separate thread."""
  register_callback(subscribe_webcam_events(path), cb)
