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

"""Scope is a module that provides visualization of data in real-time."""
import json
from typing import Any, Dict, Optional, Text, Union

import altair as alt
import pandas as pd
import six



from golang.multiscope.streams import writers_pb2 as writers_pb
from golang.multiscope.streams.tables import table_pb2 as table_pb
from golang.stream import stream_pb2 as pb
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base
from multiscope.remote.writers import utils


class DataFrameWriter(base.Writer):
  """DataFrameWriter displays a vega chart on the Multiscope page."""

  @control.init
  def __init__(self,
               name: Text,
               parent: Optional[group.ParentNode] = None,
               spec=None):


    path = group.join_path(parent, name)
    request = pb.CreateNodeRequest()
    request.path.path.extend(path)
    request.type = writers_pb.Writers().vega_writer
    create_group_resp = stream_client.CreateNode(request=request)
    super().__init__(path=tuple(create_group_resp.path.path))
    self.spec_path = self.path + ('specification',)
    self.data_path = self.path + ('data',)
    active_paths.register_callback(self.data_path, self._set_should_write)  # pytype: disable=wrong-arg-types

    self._set_display()

    # self.chart can be used to set the spec
    self.chart: Optional[alt.Chart] = None
    self._last_chart = self.chart
    if spec:
      self.set_spec(spec)


  @control.method
  def write(self, df: Union[pd.DataFrame, Dict[Any, Any]]):
    """Write a new dataframe."""
    if isinstance(df, dict):
      df = pd.DataFrame.from_dict(df)
    assert len(df.shape) == 2

    if self._last_chart is not self.chart:
      if self.chart is not None:
        self.set_spec(self.chart.to_json())
      self._last_chart = self.chart

    table = table_pb.Table()
    table.length = df.shape[0]  # number of rows
    for column_name in df.columns:
      column = table_pb.Column(name=column_name)
      if pd.api.types.is_numeric_dtype(df[column_name].dtype):
        column.float_data.extend(df[column_name].values)
      else:
        column.string_data.extend(df[column_name].astype(str).values)
      table.columns.append(column)

    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.data_path)
    request.data.pb.Pack(table)
    stream_client.PutNodeData(request=request)

  @property
  @control.method
  def altair_data(self):
    return alt.Data(values=[])

  @control.method
  def set_spec(self, spec):
    """Set the vega specification of the chart."""
    height, width = utils.extract_size(spec)
    if not isinstance(spec, six.string_types):
      spec = json.dumps(spec)
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.spec_path)
    request.data.raw = spec.encode('utf-8')
    stream_client.PutNodeData(request=request)
    self.set_display_size(height=height, width=width)
