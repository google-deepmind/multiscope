from typing import Any, NamedTuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from multiscope.protos import ticker_pb2
from multiscope.protos import ticker_pb2_grpc
from multiscope.protos import tree_pb2 as pb
from multiscope import remote as multiscope
from multiscope.remote import clock
from multiscope.remote import stream_client


class FakeTickerResponse(NamedTuple):
    ticker: Any


def _to_event_pb(path, msg):
    event = pb.Event()
    event.path.path.extend(path)
    event.payload.Pack(msg)
    return event


def _send_event(event):
    """Sends the event."""
    req = pb.SendEventsRequest()
    req.events.append(event)
    stream_client.SendEvents(req)


def setUpModule():
    multiscope.start_server()


class TickerTest(parameterized.TestCase):
    def testCounter(self):
        """Asserts the tick number goes up."""

        ticker = clock.Ticker("ticker")
        self.assertEqual(0, ticker.current_tick)
        ticker.tick()
        self.assertEqual(1, ticker.current_tick)
        ticker.tick()
        self.assertEqual(2, ticker.current_tick)

    @parameterized.parameters([0, 10, 100])
    def testSetPeriod(self, period_ms):
        """Tests that we can set the period from the server."""
        ticker = clock.Ticker("ticker")
        ticker._register_event_listener(
            lambda a: self.assertAlmostEqual(period_ms, ticker.period)
        )

        action = ticker_pb2.TickerAction()
        action.setPeriod.period_ms = period_ms
        _send_event(_to_event_pb(ticker.path, action))

    @mock.patch.object(ticker_pb2_grpc, "TickersStub")
    def testWrite(self, mock_tickers_stub):
        # Mocks and dynamically constructed interfaces (which protos are in python)
        # don't play super nice together. See http://go/python-tips/049 for tips
        # on this.
        pb_ticker = ticker_pb2.Ticker()
        pb_ticker.path.path.extend(["ticker"])
        mock_write = mock.MagicMock()
        mock_tickers_stub.return_value.New.return_value = FakeTickerResponse(
            ticker=pb_ticker
        )
        mock_tickers_stub.return_value.Write = mock_write

        ticker = clock.Ticker("ticker")
        ticker.tick()
        mock_write.assert_called()


class TimerTest(parameterized.TestCase):
    def testDifference(self):
        limit = 0.1
        timer = clock.Timer(ema_sample_weight=1.0)
        timer.start()
        samples = [timer.sample().total_seconds() for i in range(20)]
        if any([s > limit for s in samples]):
            self.fail(
                "Some time differences are higher than the limit "
                f"({limit}): {samples}."
            )


if __name__ == "__main__":
    absltest.main()
