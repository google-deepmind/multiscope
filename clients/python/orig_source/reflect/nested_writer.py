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

"""NestedWriter is a module that provides multiscope visualization of nested tensors."""

from typing import (Any, Mapping, Optional, Text, Union)

import numpy as np
import tree

from multiscope.reflect import parsers
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote.writers import base


class NestedWriter(base.Writer):
  """NestedWriter displays tensors in nested data structures in Multiscope."""

  @control.init
  def __init__(
      self,
      name: Text,
      parent: Optional[group.ParentNode] = None,
      force_write: bool = False,
  ):
    self._group = group.Group(name, parent)
    self._mapping = dict()
    self._fixed = dict()
    self._force_write = force_write

  @property
  def should_write(self):
    return True

  @control.method
  def reset(self):
    tree.map_structure(lambda x: x.reset(), self._mapping)

  @control.method
  def write(self, data: Union[Mapping[Text, Any], Any]):
    contents = None
    if hasattr(data, 'items'):
      contents = data.items()
    elif hasattr(data, '_asdict'):
      contents = data._asdict().items()
    elif isinstance(data, tuple):
      contents = tuple(zip([str(d) for d in range(len(data))], data))
    elif isinstance(data, list):
      contents = tuple(zip([str(d) for d in range(len(data))], data))
    elif isinstance(data, np.ndarray):
      contents = ((self._group.name, data),)

    if contents is None:
      raise TypeError('Type :' + str(type(data)) +
                      ' is not a nested type (list, tuple, Mapping).')
    for name, child in contents:
      writer = self._mapping.get(name, None)
      if writer is None:
        writer = parsers.new_abstract_writer(name, self._group, child,
                                             self._force_write)
        self._mapping[name] = writer
      if not self._force_write and not writer.should_write:
        continue
      writer.write(child)
