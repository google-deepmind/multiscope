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
from typing import (
    Callable,
    Generator,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

from absl import logging

from multiscope.protos import ticker_pb2
from multiscope.protos import tree_pb2 as pb
from multiscope.remote import stream_client

_ticker_filter = ticker_pb2.TickerAction.DESCRIPTOR.full_name


class EventProcessor:
    """Process events coming from the server."""

    def __init__(self):
        self.__path_to_cb = {}
        self.__mutex = threading.Lock()
        self.__event_stream: Iterable = None
        global _event_processor

    def run(self):
        self.__event_stream = stream_client.StreamEvents(
            pb.StreamEventsRequest())
        ack_event = next(self.__event_stream)
        if ack_event.payload.type_url:
            raise ValueError(
                "the server sent an incorrect acknowledgement event")
        self.__thread = threading.Thread(
            target=self.__process,
            name="process_events",
            daemon=True,
        ).start()

    def __process(self):
        """Connect to Multiscope server to start a stream to maintain the list of active paths from client requests.

        This function is the main of demon thread updating the list of active paths
        for Multiscope with respect to how web clients connect to the Mutiscope
        server.
        """
        for event in self.__event_stream:
            key = tuple(p for p in event.path.path)
            callbacks = self.__path_to_cb.get(key, [])
            for cb in callbacks:
                cb(event)

    def register_callback(self, path: Sequence[str],
                          cb: Callable[[pb.Event], None]) -> None:
        """Calls the provided cb with every element of gen in a separate thread."""
        key = tuple(path)
        with self.__mutex:
            callbacks = self.__path_to_cb.get(key, [])
            callbacks.append(cb)
            self.__path_to_cb[key] = callbacks


_event_processor = EventProcessor()


def register_callback(path: Sequence[str], cb: Callable[[pb.Event],
                                                        None]) -> None:
    """Calls the provided cb with every element of gen in a separate thread."""
    _event_processor.register_callback(path, cb)


def register_ticker_callback(
    path: Sequence[str],
    cb: Callable[[ticker_pb2.TickerAction], None],
):
    """Calls the provided cb with every mouse event at the provided path in a separate thread."""

    def process(event: pb.Event):
        action_event = ticker_pb2.TickerAction()
        action_event.ParseFromString(event.payload.value)
        cb(action_event)

    register_callback(path, process)
