import threading
import time

from absl.testing import absltest

from multiscope.protos import ticker_pb2
from multiscope.protos import tree_pb2 as pb

from multiscope import remote as multiscope
from multiscope.remote import stream_client


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

    # Register the callback
    path = ["this", "is", "a", "path"]
    queue = multiscope.events.subscribe_ticker_events(path)

    # Create an event
    data = ticker_pb2.TickerAction()
    data.command = ticker_pb2.TickerAction.Command.RUN
    event = _to_event_pb(path, data)
    req = pb.SendEventsRequest()
    req.events.append(event)

    # Continuously send events in the background. This is necessary because the
    # server responds with the grpc unary stream straight away, but cannot tell
    # us when the subscription above is actually registered.
    def send_events():
      while True:
        stream_client.SendEvents(req)
        time.sleep(1)

    threading.Thread(
        target=send_events,
        daemon=True,
    ).start()

    # Checks that the returned event has everything we need
    ret_path, ret_payload = next(queue)
    self.assertIsNotNone(ret_payload)
    self.assertEqual(path, ret_path)
    self.assertEqual(data, ret_payload)


if __name__ == "__main__":
  absltest.main()
