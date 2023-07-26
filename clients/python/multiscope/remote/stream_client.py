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
"""A GRPC client for talking to the stream backend."""

import logging
import threading
from typing import Optional, Text
from urllib import parse

import grpc

from multiscope.protos import tree_pb2 as pb
from multiscope.protos import tree_pb2_grpc as pb_grpc
from multiscope.remote import active_paths
from multiscope.remote.events import events

_channel: Optional[grpc.Channel] = None


def InitGlobalConnection(grpc_url: Text) -> grpc.Channel:
  global _channel
  if _channel is not None:
    raise AssertionError("GRPC stub already initialized")
  creds = grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)
  _channel = grpc.secure_channel(
      grpc_url,
      credentials=creds,
      # Remove all grpc limits on max message size to support writing very
      # large messages (eg the mujoco scene init message).
      #
      # This effectively limits the message to the default protobuf max message
      # size. See also http://yaqs/5863428325507072.
      options=[
          ("grpc.max_send_message_length", -1),
          ("grpc.max_receive_message_length", -1),
      ],
  )
  return _channel


class Client:
  """Client to the Multiscope server.
  """

  def __init__(self, timeout_secs):
    self._tree_client = pb_grpc.TreeStub(_channel)
    resp = self._tree_client.GetTreeID(
        pb.GetTreeIDRequest(), wait_for_ready=True, timeout=timeout_secs)
    self._tree_id = resp.tree_id
    self._reset_epoch: int = 0
    self._mu = threading.Lock()
    # Listen to events from the server.
    self._event_processor = events.EventProcessor(self)
    self._event_processor.run()
    self._active_paths = active_paths.ActivePath(self)
    self._active_paths.run()

  def Channel(self):
    return _channel

  def TreeID(self):
    return self._tree_id

  def TreeClient(self):
    return self._tree_client

  def ResetEpoch(self) -> int:
    """Returns the number of times ResetState has been called."""
    with self._mu:
      return self._reset_epoch

  def ActivePaths(self, *args, **kwargs):
    return self._active_paths

  def ResetState(self, *args, **kwargs):
    self._reset_epoch
    with self._mu:
      res = self._tree_client.ResetState(*args, **kwargs)
      self._reset_epoch += 1
      return res

  def Events(self):
    return self._event_processor


_client: Optional[Client] = None


def Initialized() -> bool:
  return _client is not None


def SetGlobalClient(client: Client):
  global _client
  _client = client


def GlobalClient():
  """Return the global client."""
  return _client
