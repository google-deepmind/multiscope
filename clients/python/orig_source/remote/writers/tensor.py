"""Scope is a module that provides visualization of data in real-time."""
from typing import Optional, Text

import numpy as np

from golang.multiscope.streams.tensors import tensors_pb2 as tensors_pb
from golang.multiscope.streams.tensors import tensors_pb2_grpc as tensors_grpc
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base


class TensorWriter(base.Writer):
  """TensorWriter displays a vega chart on the Multiscope page."""

  @control.init
  def __init__(self, name: Text, parent: Optional[group.ParentNode] = None):

    self._client = tensors_grpc.TensorWritersStub(stream_client.channel)
    path = group.join_path_pb(parent, name)
    req = tensors_pb.NewTensorWriterRequest(path=path)
    self.writer = self._client.NewTensorWriter(req).writer
    super().__init__(path=tuple(self.writer.path.path))
    active_paths.register_callback(self.path, self._set_should_write)

    self._set_display()

  @control.method
  def write(self, tensor: np.ndarray):
    """Write a tensor."""
    req = tensors_pb.WriteTensorRequest()
    req.writer.path.path.extend(self.path)
    req.tensor.shape.extend(tensor.shape)
    if tensor.dtype is not np.float32:
      tensor = tensor.astype(np.float32)
    req.tensor.value = tensor.tobytes()
    req.tensor.dtype = tensors_pb.FLOAT32
    self._client.WriteTensor(req)

  @control.method
  def reset(self):
    req = tensors_pb.ResetTensorWriterRequest()
    req.writer.path.path.extend(self.path)
    self._client.ResetTensorWriter(req)
