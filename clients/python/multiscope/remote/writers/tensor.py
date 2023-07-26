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
"""Scope is a module that provides visualization of data in real-time."""
from typing import Any, Optional, Tuple, Type

from multiscope.protos import tensor_pb2
from multiscope.protos import tensor_pb2_grpc
from multiscope.remote import active_paths
from multiscope.remote.control import control
from multiscope.remote.control import decorators
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base
import numpy as np


class TensorWriter(base.Writer):
  """TensorWriter displays a vega chart on the Multiscope page."""

  @decorators.init
  def __init__(
      self,
      py_client: stream_client.Client,
      name: str,
      parent: Optional[group.ParentNode] = None,
  ):
    self._client = tensor_pb2_grpc.TensorsStub(py_client.Channel())
    path = group.join_path_pb(parent, name)
    req = tensor_pb2.NewWriterRequest(tree_id=py_client.TreeID(), path=path)
    self.writer = self._client.NewWriter(req).writer
    super().__init__(py_client=py_client, path=tuple(self.writer.path.path))
    self.register_activity_callback(self._set_should_write)

  @decorators.method
  def write(self, tensor: np.ndarray):
    """Write a tensor."""
    req = tensor_pb2.WriteRequest(writer=self.writer)
    tensor, dtype = _normalize_tensor_dtype(tensor)
    req.tensor.dtype = dtype
    _add_tensor_shape(req.tensor, tensor.shape)
    req.tensor.content = tensor.tobytes()
    self._client.Write(req)

  @decorators.method
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
