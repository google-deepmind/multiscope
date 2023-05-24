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

_event_processor = None


class EventProcessor:
    """Process events coming from the server."""

    def __init__(self):
        self.__path_to_cb = {}
        self.__mutex = threading.Lock()
        global _event_processor

    def run(self):
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
        events = stream_client.StreamEvents(pb.StreamEventsRequest())
        for event in events:
            path = tuple(p for p in event.path.path)
            callbacks = self.__path_to_cb.get(path, [])
            for cb in callbacks:
                cb(event)

    def register_callback(
        self, path: Sequence[str], cb: Callable[pb.Event, None]
    ) -> None:
        """Calls the provided cb with every element of gen in a separate thread."""
        with self.__mutex:
            callbacks = self.__path_to_cb.get(path, [])
            callbacks.append(cb)
            self.__path_to_cb[tuple(path)] = callbacks


def register_callback(path: Sequence[str], cb: Callable[pb.Event, None]) -> None:
    """Calls the provided cb with every element of gen in a separate thread."""
    _event_processor.register_callback(path, cb)


def register_ticker_callback(
    cb: Callable[ticker_pb2.TickerAction, None],
    path: Sequence[str],
):
    """Calls the provided cb with every mouse event at the provided path in a separate thread."""

    def process(event: pb.Event):
        action_event = ticker_pb2.TickerAction()
        action_event.ParseFromString(event.payload.value)
        cb(action_event)

    register_callback(path, process)
