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

"""A GRPC client for talking to the stream backend."""

import logging
import threading
from typing import Optional, Text
from urllib import parse

import grpc

from multiscope.protos import tree_pb2 as pb
from multiscope.protos import tree_pb2_grpc as pb_grpc

_stub: Optional[pb_grpc.TreeStub] = None
_reset_epoch: int = 0
_mu = threading.Lock()

channel: Optional[grpc.Channel] = None


def InitializeStub(url: Text) -> None:
    """Initializes the connection to the multiscope server."""
    global _stub
    if _stub is not None:
        raise AssertionError("GRPC stub already initialized")

    global channel
    creds = grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)
    channel = grpc.secure_channel(
        url,
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
    _stub = pb_grpc.TreeStub(channel)


def TryConnecting(timeout_secs: int):
    _stub.GetNodeData(
        pb.NodeDataRequest(reqs=[]), wait_for_ready=True, timeout=timeout_secs
    )


def Initialized() -> bool:
    return _stub is not None


def ResetEpoch() -> int:
    """Returns the number of times ResetState has been called."""
    with _mu:
        return _reset_epoch


def GetNodeStruct(*args, **kwargs):
    return _stub.GetNodeStruct(*args, **kwargs)


def GetNodeData(*args, **kwargs):
    return _stub.GetNodeData(*args, **kwargs)


def StreamEvents(*args, **kwargs):
    return _stub.StreamEvents(*args, **kwargs)


def SendEvents(*args, **kwargs):
    return _stub.SendEvents(*args, **kwargs)


def ActivePaths(*args, **kwargs):
    return _stub.ActivePaths(*args, **kwargs)


def ResetState(*args, **kwargs):
    global _reset_epoch
    with _mu:
        res = _stub.ResetState(*args, **kwargs)
        _reset_epoch += 1
        return res
