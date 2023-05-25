"""Scope is a module that provides visualization of data in real-time."""
from typing import Any, Optional, Tuple, Type

import numpy as np

from multiscope.protos import tensor_pb2
from multiscope.protos import tensor_pb2_grpc

from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base


class TensorWriter(base.Writer):
  """TensorWriter displays a vega chart on the Multiscope page."""

  @control.init
  def __init__(self, name: str, parent: Optional[group.ParentNode] = None):

    self._client = tensor_pb2_grpc.TensorsStub(stream_client.channel)
    path = group.join_path_pb(parent, name)
    req = tensor_pb2.NewWriterRequest(path=path)
    self.writer = self._client.NewWriter(req).writer
    super().__init__(path=tuple(self.writer.path.path))
    active_paths.register_callback(self.path, self._set_should_write)

    # TODO(b/251324180): re-enable once fixed.
    # self._set_display()

  @control.method
  def write(self, tensor: np.ndarray):
    """Write a tensor."""
    req = tensor_pb2.WriteRequest()
    req.writer.path.path.extend(self.path)
    tensor, dtype = _normalize_tensor_dtype(tensor)
    req.tensor.dtype = dtype
    _add_tensor_shape(req.tensor, tensor.shape)
    req.tensor.content = tensor.tobytes()
    self._client.Write(req)

  @control.method
  def reset(self):
    req = tensor_pb2.ResetWriterRequest()
    req.writer.path.path.extend(self.path)
    self._client.ResetWriter(req)


def _normalize_tensor_dtype(tensor: np.ndarray,) -> Tuple[np.ndarray, Any]:
  """Converts the type into the closest of tensor_pb2.DataType."""
  # TODO: can't seem to mark that the second entry is tensor_pb2.DataType
  # in the type annotation.
  if tensor.dtype is np.uint8:
    return tensor, tensor_pb2.DataType.DT_UINT8

  if tensor.dtype is not np.float32:
    tensor = tensor.astype(np.float32)

  return tensor, tensor_pb2.DataType.DT_FLOAT32


def _add_tensor_shape(tensor: tensor_pb2.Tensor, shape: Tuple[int,
                                                              ...]) -> None:
  """Modifies the tensor in place to add the shape."""
  for axis_len in shape:
    tensor.shape.dim.append(tensor_pb2.Shape.Dim(size=axis_len))
