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

"""Define basic types for parsers."""

import abc
from typing import NamedTuple
import six

from multiscope.reflect import updater
from multiscope.remote import group


class Parent(NamedTuple):
  node: group.ParentNode
  obj: updater.ReflectTarget


class State:
  """State contains the state of the parser during the exploration."""

  def __init__(self, parsers, ticker, default_parser):
    self.ticker = ticker
    self.updater = updater.Updater()
    self._parents = [Parent(ticker, updater.ReflectTarget(None))]
    self._parsers = parsers
    self._default_parser = default_parser

  def has_parent(self, obj: updater.ReflectTarget) -> bool:
    for parent in self._parents:
      if parent.obj is obj:
        return True
    return False

  def push_parent(self, parent: Parent):
    return self._parents.append(parent)

  def pop_parent(self) -> Parent:
    return self._parents.pop()

  def find_parser(self, obj):
    for candidate_parser in self._parsers:
      if candidate_parser.can_parse(obj):
        return candidate_parser
    return self._default_parser

  @property
  def parent(self) -> Parent:
    return self._parents[-1]


@six.add_metaclass(abc.ABCMeta)
class Parser:
  """Parser constructs Multiscope parsers for a given Python type.

  A parser should have no state.
  """

  @abc.abstractmethod
  def can_parse(self, obj: updater.ReflectTarget):
    """Returns true if this parser can parse `obj`.

    Args:
      obj: An object to build a Multiscope tree for.
    """

  @abc.abstractmethod
  def parse(self, state: State, name: str, obj: updater.ReflectTarget):
    """Build a subtree to represent `obj` under the parent node `parent`.

    Args:
       state: Current state of the parser.
       name: The name of the variable to write.
       obj: An instance of the variable to write.
    """

  @abc.abstractmethod
  def new_abstract_writer(self, name: str, parent_node: group.ParentNode,
                          obj: updater.ReflectTarget, force_write: bool):
    raise NotImplementedError
