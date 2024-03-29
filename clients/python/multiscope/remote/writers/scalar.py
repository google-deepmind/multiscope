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
"""Writes scalars to Multiscope."""
from typing import Any, Mapping, Optional

from multiscope.protos import scalar_pb2
from multiscope.protos import scalar_pb2_grpc
from multiscope.remote.control import control
from multiscope.remote.control import decorators
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base


class ScalarWriter(base.Writer):
  """Display a time series plot on the Multiscope page.

  This Writer writes dictionaries of floats or sequences of floats. Example:

  ```
  scalar_writer.write({
      'b': 5.7,
      'a': [1, 2, 3],
  })
  ```

  will result in four entries, `'b': 5.7, 'a1':1, 'a2':2, 'a3':3`,
  on multiscope.
  """

  @decorators.init
  def __init__(
      self,
      py_client: stream_client.Client,
      name: str,
      parent: Optional[group.ParentNode] = None,
  ):
    self._client = scalar_pb2_grpc.ScalarsStub(py_client.Channel())
    path = group.join_path_pb(parent, name)
    req = scalar_pb2.NewWriterRequest(tree_id=py_client.TreeID(), path=path)
    self.writer = self._client.NewWriter(req).writer

    super().__init__(py_client=py_client, path=tuple(self.writer.path.path))
    self.register_activity_callback(self._set_should_write)

  @decorators.method
  def write(self, values: Mapping[str, Any]):
    """Writes data to the vega writer."""
    converted_values: dict[str, float] = {}
    for k, v in values.items():
      try:
        for sub_key, sub_val in enumerate(v):
          converted_values[k + str(sub_key)] = float(sub_val)
      except TypeError:
        converted_values[k] = float(v)

    if not converted_values:
      return

    request = scalar_pb2.WriteRequest(
        writer=self.writer,
        label_to_value=converted_values,
    )
    self._client.Write(request)
