"""A Multiscope player to play back data."""
from typing import Any, Callable, List, Optional

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


class Player(group.ParentNode):
  """An object that can play back data.

    A `Player` enables a panel in Multiscope with controls to play back the data.
    The client needs to call `store` which will cause the server to save all the children
    of the node into a frame.

    The UI will provide controls to display frames back in time.
    """

  @control.init
  def __init__(
      self,
      name: str,
      parent: Optional[group.ParentNode] = None,
      stoppable: bool = True,
  ):
    self._tick_num: int = 0

    # Make the connection to the multiscope server.
    self._client = ticker_pb2_grpc.TickersStub(stream_client.channel)
    path = group.join_path_pb(parent, name)
    req = ticker_pb2.NewPlayerRequest(path=path, ignorePause=not stoppable)
    self._player = self._client.NewPlayer(req).player
    super().__init__(path=tuple(self._player.path.path))

  def store_frame(self) -> None:
    """Store all children nodes in storage."""
    data = ticker_pb2.PlayerData(tick=self._tick_num)
    req = ticker_pb2.StoreFrameRequest()
    req.player.CopyFrom(self._player)
    req.data.CopyFrom(data)
    self._client.StoreFrame(req)
