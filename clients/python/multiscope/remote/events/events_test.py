import threading
import time

from absl.testing import absltest

from multiscope.protos import ticker_pb2
from multiscope.protos import tree_pb2 as pb

from multiscope.remote import server as multiscope
from multiscope.remote import stream_client
from multiscope.remote.events import events


def _to_event_pb(path, msg):
  event = pb.Event()
  event.path.path.extend(path)
  event.payload.Pack(msg)
  return event


class EventsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    multiscope.start_server()

  def testSubscribe(self):
    """Tests if we can subscribe to events."""

    wait = threading.Lock()
    wait.acquire()

    collected_action = None

    def collect_event(action):
      nonlocal collected_action
      collected_action = action
      wait.release()

    # Register the callback
    path = ["this", "is", "a", "path"]
    events.register_ticker_callback(path, collect_event)
    # Create an event
    data = ticker_pb2.TickerAction(command=ticker_pb2.Command.CMD_RUN)
    event = _to_event_pb(path, data)
    req = pb.SendEventsRequest()
    req.events.append(event)

    stream_client.SendEvents(req)

    wait.acquire()
    self.assertIsNotNone(collected_action)
    self.assertEqual(data, collected_action)


if __name__ == "__main__":
  absltest.main()
