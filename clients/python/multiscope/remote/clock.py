"""A Multiscope clock/ticker for synchronizing writes."""
from typing import Any, Callable, List, Optional, Sequence, Text

import datetime
import time
import timeit
import threading
from absl import logging

from google.protobuf import duration_pb2

from multiscope.protos import ticker_pb2
from multiscope.protos import ticker_pb2_grpc
from multiscope.protos import tree_pb2 as pb
from multiscope.remote import control
from multiscope.remote import events
from multiscope.remote import group
from multiscope.remote import stream_client


TIMER_AVERAGE_STEPSIZE = 0.01


# TODO: this used to be called Clock. Rename everywhere.
class Ticker(group.ParentNode):
    """An object that can perform synchronization.

    A `Ticker` synchronizes the calls of all writers in its thread. It can

    * ensure that all writes happen with a given period, and
    * pause the execution of the thread it is in.

    The ticker can be controlled either through its methods or through its panel
    on the multiscope web interface.

    ## Thread-safety

    Since we can control the ticker from the UI (or locally) some of the
    methods and properties here are thread-safe:

    * period (reading and writing),
    * pause,
    * run,
    * step.
    """

    @control.init
    def __init__(self, name: Text, parent: Optional[group.ParentNode] = None):
        self._tick_num: int = 0
        # listeners are called on each tick.
        self._listeners: Sequence[Callable[[], None]] = []

        # Use timers to ensure the correct period and collect stats.
        self._tick_timer = _Timer(TIMER_AVERAGE_STEPSIZE)

        # Make the connection to the multiscope server.
        self._client = ticker_pb2_grpc.TickersStub(stream_client.channel)
        path = group.join_path_pb(parent, name)
        req = ticker_pb2.NewTickerRequest(path=path)
        self._ticker = self._client.New(req).ticker
        super().__init__(path=tuple(self._ticker.path.path))

        # Set up event management.
        events.register_ticker_callback(cb=self._process_event, path=self.path)
        # Notify listeners when an event happens. Primarily used for testing.
        self._event_listeners: List[Callable[[ticker_pb2.TickerAction], Any]] = []

        # Setup for controlling the ticker across threads.
        # `set()` the event to resume execution when paused.
        self._resume = threading.Event()
        self._pause_next_step = False
        self._pause_lock = threading.Lock()
        self._period: datetime.timedelta = datetime.timedelta()
        self._period_lock = threading.Lock()

        # Schedule the writing of statistics with each tick.
        self.register_listener(self._write_data)

    def _call_listeners(self) -> None:
        """Calls all listeners."""
        # TODO: collect statistics about how long these calls take.
        # TODO: collect and surface errors from these calls in a good way.
        for fn in self._listeners:
            fn()

    def _write_data(self) -> None:
        """Writes `TickerData`."""
        data = ticker_pb2.TickerData()
        data.tick = self._tick_num
        # TODO: write real data.
        data.periods.total.CopyFrom(duration_pb2.Duration(seconds=12))
        data.periods.experiment.CopyFrom(duration_pb2.Duration(seconds=1))
        data.periods.callbacks.CopyFrom(duration_pb2.Duration(seconds=4))
        data.periods.idle.CopyFrom(duration_pb2.Duration(seconds=10))
        req = ticker_pb2.WriteRequest()
        req.ticker.CopyFrom(self._ticker)
        req.data.CopyFrom(data)
        self._client.Write(req)

    def tick(self) -> None:
        """Waits until it's time to continue running."""
        since_last_tick = self._tick_timer.sample()

        if since_last_tick < self._period:
            time.sleep((self._period - since_last_tick).total_seconds)

        with self._pause_lock:
            should_pause = self._pause_next_step

        if should_pause:
            self._resume.wait()
            self._resume.clear()

        self._tick_timer.start()  # Measure the period from here.
        # Note: the go implementation only starts measuring time after calling
        # the listeners.
        self._tick_num += 1
        self._call_listeners()

    @control.method
    def register_listener(self, fn: Callable[[], None]):
        """Registers `fn` to be called everytime Tick is called."""
        self._listeners.append(fn)

    @property
    @control.method
    def current_tick(self) -> int:
        """Returns the current tick of the clock."""
        return self._tick_num

    # @control.method
    def _register_event_listener(self, fn: Callable[[], None]):
        """Registers `fn` to be called everytime an event is received."""
        self._event_listeners.append(fn)

    def _process_event(self, path, action: ticker_pb2.TickerAction) -> None:
        """Processes events from the multiscope server."""
        del path  # Could verify this.
        action_type = action.WhichOneof("action")
        # This implementation executes each command. An alternative is to let
        # them potentially queue up, ignore all but the last one of each type.
        if action_type == "setPeriod":
            # Calling it through the property setter is thread-safe.
            self.period = float(action.setPeriod.period_ms) * 1e-6
        elif action_type == "command":
            self._process_control_cmd(action.command)
        else:
            raise ValueError("Unexpected `action` of `TickerAction`.")

        for fn in self._event_listeners:
            fn(action)

    def _process_control_cmd(self, command: ticker_pb2.TickerAction.Command) -> None:
        if command == ticker_pb2.TickerAction.Command.NONE:
            logging.info("Received a NONE TickerAction Command.")
        elif command == ticker_pb2.TickerAction.Command.STEP:
            self.step()
        elif command == ticker_pb2.TickerAction.Command.PAUSE:
            self.pause()
        elif command == ticker_pb2.TickerAction.Command.RUN:
            self.run()
        else:
            raise ValueError(
                "Unexpected value of enum TickerAction.Command: {command}."
            )

    @property
    @control.method
    def period(self) -> float:
        """Gets the period of the clock in milliseconds. Thread-safe."""
        with self._period_lock:
            return self._period.total_seconds() * 1000

    @period.setter
    @control.method
    def period(self, p_nanos: int):
        """Sets the period in nano seconds. Thread-safe."""
        period = datetime.timedelta(milliseconds=p_nanos * 1e6)
        with self._period_lock:
            self._period = period

    @control.method
    def step(self):
        """Uses the ticker into step-by-step mode. Thread-safe.

        If the ticker is currently running, this will pause it. If it is
        already paused, this will do exactly one stop then pause again.
        """
        logging.debug("CMD: step.")
        with self._pause_lock:
            self._pause_next_step = True
        self._resume.set()

    @control.method
    def pause(self):
        """Pauses the clock on the next tick. Thread-safe."""
        logging.debug("CMD: pause.")
        with self._pause_lock:
            self._pause_next_step = True

    @control.method
    def run(self):
        """Runs the clock if paused, otherwise is a no-op. Thread-safe."""
        logging.debug("CMD: run.")
        with self._pause_lock:
            self._pause_next_step = False
        self._resume.set()


class _Timer:
    """Measures time between samples. Can provide an EMA average of it."""

    def __init__(self, ema_sample_weight: float):
        self._last_sample: Optional[timeit.Timer] = None
        self._ema_sample_weight = ema_sample_weight
        self._average: float = 0.0  # TODO: there are smarter ways to set this.

    def sample(self) -> datetime.timedelta:
        """Records and returns the time since the last sample."""
        if self._last_sample is None:
            self.start()
            return datetime.timedelta()

        cur_time = timeit.default_timer()
        diff = datetime.timedelta(cur_time - self._last_sample)
        self._last_sample = cur_time

        self._average = (
            self._ema_sample_weight * diff.total_seconds()
            + (1.0 - self._ema_sample_weight) * self._average
        )

        return diff

    def start(self):
        """The next sample will measure the time from this point."""
        self._last_sample = timeit.default_timer()

    @property
    def average(self):
        return datetime.timedelta(self._average)
