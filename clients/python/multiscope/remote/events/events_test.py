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
