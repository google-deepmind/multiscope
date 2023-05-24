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

"""Parser for tensor types (np.ndarray). If 1D Array then use a ScalarWriter."""

import numpy as np

from multiscope.reflect import parser
from multiscope.reflect import parsers
from multiscope.reflect import updater
from multiscope.remote import group
from multiscope.remote.writers import scalar
from multiscope.remote.writers import tensor


class TensorParser(parser.Parser):
  """Parse numerical values."""

  def can_parse(self, obj):
    """Returns true if this parser can parse `obj`.

    Args:
      obj: An object to build a Multiscope tree for.
    """
    return isinstance(obj, np.ndarray)

  def parse(self, state: parser.State, name: str, obj: updater.ReflectTarget):
    """Build a subtree to represent `obj` under the parent node `parent`.

    In this parser, obj (a numpy array) is ignored because the data puller will
    use getattr(parent.obj, name) to get the current reference to the numpy
    array.

    Args:
       state: Current state of the parser.
       name: The name of the variable to write.
       obj: An instance of the variable to write.
    """
    parent = state.parent
    wrt = tensor.TensorWriter(name, parent.node)
    data_puller = updater.ObjectDataPuller(wrt, name, parent.obj)
    wrt.register_activity_callback(state.updater.new_callback(data_puller))

  def new_abstract_writer(self, name: str, parent_node: group.ParentNode,
                          obj: np.ndarray, force_write: bool):
    if obj.ndim == 0:
      return parsers.ConcreteWriter(
          writer=scalar.ScalarWriter(name, parent_node),
          write_function=lambda x: {'value': x.item()},
          force_write=force_write)
    else:
      return parsers.ConcreteWriter(
          writer=tensor.TensorWriter(name, parent_node),
          write_function=lambda x: x,
          force_write=force_write)
