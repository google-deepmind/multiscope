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
    # TODO: Use authenticated channels.
    global channel
    logging.warning("Using an unsecure gRPC channel!")
    channel = grpc.insecure_channel(url)
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
