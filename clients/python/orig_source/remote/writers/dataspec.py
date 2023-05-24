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
import numbers
from typing import Any, Dict, List, Optional, Text, Union

import altair as alt
import numpy as np



from golang.multiscope.streams import writers_pb2 as writers_pb
from golang.stream import stream_pb2 as pb
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.writers import base
from multiscope.remote.writers import utils


class DataSpecWriter(base.Writer):
  """DataSpecWriter displays a vega chart on the Multiscope page."""

  @control.init
  def __init__(self,
               name: Text,
               parent: Optional[group.ParentNode] = None,
               spec=None):


    path = group.join_path(parent, name)
    # Create a Group node at self.path
    request = pb.CreateNodeRequest()
    request.path.path.extend(path)
    request.type = writers_pb.Writers().vega_writer
    create_group_resp = stream_client.CreateNode(request=request)
    super().__init__(path=tuple(create_group_resp.path.path))
    self.spec_path = self.path + ('specification',)
    self.data_path = self.path + ('data',)
    active_paths.register_callback(self.data_path, self._set_should_write)  # pytype: disable=wrong-arg-types

    self._set_display()

    self._spec: Optional[Dict[Text, Any]] = None
    if spec:
      self.set_spec(spec)


  @control.method
  def write(self, data_spec: Union[Dict[Text, Any], List[Dict[Text, Any]]]):
    """Writes a data specification, merging it with any existing one.

    Args:
      data_spec: The new data specification.

    The specification may be a single dict such as:
      {"name": "my_dataset", "values": [...]}
    or an array of dicts for multiple datasets:
      [{"name": "data1", "values": [...]}, {"name": "data2", "values": [...]}].
    """
    if not isinstance(data_spec, list):
      data_spec = [data_spec]

    if self._spec is None:
      return

    new_data = to_json_value(data_spec)

    # Merge data into spec, overwriting old datasets and appending new ones.
    data_indices = {
        data['name']: i for i, data in enumerate(self._spec['data'])
    }
    for data in new_data:
      name = data['name']
      if name in data_indices:
        self._spec['data'][data_indices[name]] = data
      else:
        self._spec['data'].append(data)
    self.set_spec(self._spec)

  @property
  @control.method
  def altair_data(self):
    return alt.Data(values=[])

  @control.method
  def set_spec(self, spec):
    """Set the vega specification of the chart."""
    height, width = utils.extract_size(spec)
    if isinstance(spec, dict):
      spec_dict = spec
    elif isinstance(spec, str):
      spec_dict = json.loads(spec)
    else:
      raise ValueError('spec should be a valid JSON string or a dict')
    if 'data' not in spec_dict:
      spec_dict['data'] = []
    elif not isinstance(spec_dict['data'], list):
      spec_dict['data'] = [spec_dict['data']]
    self._spec = spec_dict

    spec = json.dumps(spec_dict)
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.spec_path)
    request.data.raw = spec.encode('utf-8')
    stream_client.PutNodeData(request=request)
    if width is not None and height is not None:
      self.set_display_size(height=height, width=width)


def convert_data_dict(spec: Dict[Text, Any]):
  """Converts a spec dict into JSON."""
  converted_values = {}
  for k, v in spec.items():
    converted_values[k] = to_json_value(v)
  return converted_values


def to_json_value(val):
  """Converts a single value into JSON."""
  if isinstance(val, dict):
    return convert_data_dict(val)
  elif isinstance(val, list) or isinstance(val, np.ndarray):
    for i, entry in enumerate(val):
      val[i] = to_json_value(entry)
    return val
  elif isinstance(val, str):
    return val
  elif isinstance(val, numbers.Number):
    fval = float(val)
    return str(fval) if np.isnan(val) or np.isinf(val) else fval
  else:
    raise ValueError(f'Unknown type {type(val)} contained in data spec!')
