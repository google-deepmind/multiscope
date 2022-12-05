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

from multiscope.protos import ticker_pb2
from multiscope.protos import tree_pb2 as pb
from multiscope.remote import stream_client

# _mouse_filter = userinputs_pb2.MouseEvent.DESCRIPTOR.full_name
# _keyboard_filter = userinputs_pb2.KeyboardEvent.DESCRIPTOR.full_name
_ticker_filter = ticker_pb2.TickerAction.DESCRIPTOR.full_name

_Path = Sequence[Text]
_EventPayload = Tuple[_Path, bytes]
# _MousePayload = Tuple[_Path, userinputs_pb2.MouseEvent]
# _KeyboardPayload = Tuple[_Path, userinputs_pb2.KeyboardEvent]
_TickerPayload = Tuple[_Path, ticker_pb2.TickerAction]

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


def subscribe_ticker_events(
    path: Sequence[Text]) -> EventSubscription[_TickerPayload]:
  """Returns a blocking generator of mouse events for the given path."""

  def process(payload: _EventPayload) -> _TickerPayload:
    event = ticker_pb2.TickerAction()
    event.ParseFromString(payload[1])
    return payload[0], event

  return EventSubscription(
      path=path, type_url_filter=_ticker_filter, process=process)


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


def register_ticker_callback(cb: Callable[[_TickerPayload], None],
                            path: Optional[_Path] = None):
  """Calls the provided cb with every mouse event at the provided path in a separate thread."""
  if path is None:
    path = []
  register_callback(subscribe_ticker_events(path), cb)
