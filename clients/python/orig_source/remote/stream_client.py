"""A GRPC client for talking to the stream backend."""
import threading
from typing import Optional, Text
from urllib import parse

import grpc

from golang.stream import stream_pb2 as pb
from golang.stream import stream_pb2_grpc as pb_grpc

_stub: Optional[pb_grpc.StreamStub] = None
_reset_epoch: int = 0
_mu = threading.Lock()

channel: Optional[grpc.Channel] = None


def InitializeStub(url: Text) -> None:
  """Initializes the connection to the multiscope server."""
  global _stub
  if _stub is not None:
    raise AssertionError('GRPC stub already initialized')
  if parse.urlparse(url).scheme == 'unix':
    creds = grpc.local_channel_credentials(grpc.LocalConnectionType.UDS)
  else:
    raise ValueError(
        'The selected GRPC authentication scheme not supported. Use a '
        'different url type.')
  global channel
  channel = grpc.secure_channel(
      url,
      credentials=creds,
      # Remove all grpc limits on max message size to support writing very
      # large messages (eg the mujoco scene init message).
      #
      # This effectively limits the message to the default protobuf max message
      # size. See also http://yaqs/5863428325507072.
      options=[('grpc.max_send_message_length', -1),
               ('grpc.max_receive_message_length', -1)])
  _stub = pb_grpc.StreamStub(channel)


def TryConnecting(timeout_secs: int):
  _stub.GetNodeData(
      pb.NodeDataRequest(paths=[]), wait_for_ready=True, timeout=timeout_secs)


def Initialized() -> bool:
  return _stub is not None


def ResetEpoch() -> int:
  """Returns the number of times ResetState has been called."""
  with _mu:
    return _reset_epoch


def CreateNode(*args, **kwargs):
  return _stub.CreateNode(*args, **kwargs)


def PutNodeData(*args, **kwargs):
  return _stub.PutNodeData(*args, **kwargs)


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
