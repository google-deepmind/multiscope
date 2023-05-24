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

"""Parser for numerical types (float, int)."""

import numbers

from multiscope.reflect import parser
from multiscope.reflect import parsers
from multiscope.reflect import updater
from multiscope.remote import group
from multiscope.remote.writers import scalar


class _NumListener(updater.DataPuller):
  """Pull scalar data from an object and writes to a ScalarWriter."""

  def __init__(self, writer: scalar.ScalarWriter, name: str,
               obj: updater.ReflectTarget):
    self.writer = writer
    self.name = name
    self.obj = obj

  def pull_write(self):
    self.writer.write({self.name: getattr(self.obj, self.name)})

  def reset(self):
    self.writer.reset()


class NumParser(parser.Parser):
  """Parse numerical values."""

  def can_parse(self, obj: updater.ReflectTarget):
    """Returns true if this parser can parse `obj`.

    Args:
      obj: An object to build a Multiscope tree for.
    """
    return isinstance(obj, numbers.Number)

  def parse(self, state: parser.State, name: str, obj: updater.ReflectTarget):
    """Build a subtree to represent `obj` under the parent node `parent`.

    Args:
       state: Current state of the parser.
       name: The name of the variable to write.
       obj: An instance of the variable to write.
    """
    parent = state.parent
    wrt = scalar.ScalarWriter(name, parent.node)
    data_puller = _NumListener(wrt, name, parent.obj)
    wrt.register_activity_callback(state.updater.new_callback(data_puller))

  def new_abstract_writer(self, name: str, parent_node: group.ParentNode,
                          obj: updater.ReflectTarget, force_write: bool):
    wrt = scalar.ScalarWriter(name, parent_node)
    return parsers.ConcreteWriter(wrt, lambda x: {'value': x}, force_write)
