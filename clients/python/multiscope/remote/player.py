# Copyright 2023 DeepMind Technologies Limited
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
"""A Multiscope player to play back data."""
import datetime
import threading
import time
import timeit
from typing import Any, Callable, List, Optional
from absl import logging
from google.protobuf import duration_pb2
from multiscope.protos import ticker_pb2
from multiscope.protos import ticker_pb2_grpc
from multiscope.protos import tree_pb2 as pb
from multiscope.remote.control import control
from multiscope.remote.control import decorators
from multiscope.remote.events import events
from multiscope.remote import group
from multiscope.remote import stream_client


class Player(group.ParentNode):
  """An object that can play back data.

  A `Player` enables a panel in Multiscope with controls to play back the data.
  The client needs to call `store` which will cause the server to save all the
  children
  of the node into a frame.

  The UI will provide controls to display frames back in time.
  """

  @decorators.init
  def __init__(
      self,
      py_client: stream_client.Client,
      name: str,
      parent: Optional[group.ParentNode] = None,
      stoppable: bool = True,
  ):
    self._py_client = py_client

    self._tick_num: int = 0

    # Make the connection to the multiscope server.
    self._client = ticker_pb2_grpc.TickersStub(py_client.Channel())
    path = group.join_path_pb(parent, name)
    req = ticker_pb2.NewPlayerRequest(
        tree_id=self._py_client.TreeID(), path=path, ignorePause=not stoppable)
    self._player = self._client.NewPlayer(req).player
    super().__init__(py_client=py_client, path=tuple(self._player.path.path))

  def store_frame(self) -> None:
    """Store all children nodes in storage."""
    data = ticker_pb2.PlayerData(tick=self._tick_num)
    req = ticker_pb2.StoreFrameRequest()
    req.player.CopyFrom(self._player)
    req.data.CopyFrom(data)
    self._client.StoreFrame(req)
