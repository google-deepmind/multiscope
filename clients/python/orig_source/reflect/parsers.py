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

"""Module to use reflection to automatically create Multiscope lazy writers."""

import abc
import re
from typing import List

from multiscope.reflect import class_parser
from multiscope.reflect import nested_parser
from multiscope.reflect import num_parser
from multiscope.reflect import parser
from multiscope.reflect import tensor_parser
from multiscope.reflect import text_parser
from multiscope.reflect import updater
from multiscope.remote import clock
from multiscope.remote import group
from multiscope.remote.writers import base
from multiscope.remote.writers.text import TextWriter

_parsers = [
    text_parser.TextParser(),
    tensor_parser.TensorParser(),
    num_parser.NumParser(),
    nested_parser.NestedParser(),
]


class AbstractWriter(abc.ABC):

  def write(self, obj: updater.ReflectTarget):
    raise NotImplementedError

  @property
  def should_write(self) -> bool:
    raise NotImplementedError


class ConcreteWriter(AbstractWriter):
  """An implementation of AbstractWriter to allow different types to be written."""

  def __init__(self, writer: base.Writer, write_function, force_write: bool):
    self._writer = writer
    self._write_function = write_function
    self._force_write = force_write

  @property
  def should_write(self):
    return self._force_write or self._writer.should_write

  def write(self, obj: updater.ReflectTarget):
    self._writer.write(self._write_function(obj))

  def reset(self):
    self._writer.reset()


class UnsupportedTypeParser:
  """A parser for all types that cannot be written, where an error is written."""

  def can_parse(self, _: updater.ReflectTarget):
    return True

  def parse(self, unused_state: updater.ReflectTarget, unused_name: str,
            unused_obj: updater.ReflectTarget):
    return None

  def new_abstract_writer(self, name, unused_parent_node, obj,
                          force_write: bool) -> AbstractWriter:
    text_writer = TextWriter(name)
    output = 'unhandled type: %s' % str(type(obj))
    return ConcreteWriter(
        text_writer, lambda x, out=output: out, force_write=force_write)


def new_abstract_writer(name: str, parent_node: group.ParentNode,
                        obj: updater.ReflectTarget,
                        force_write: bool) -> AbstractWriter:
  state = parser.State(
      _parsers, parent_node, default_parser=UnsupportedTypeParser())
  selected_parser = state.find_parser(obj)
  return selected_parser.new_abstract_writer(name, parent_node, obj,
                                             force_write)


def reflect(ticker: clock.Ticker, name: str, obj):
  state = parser.State(
      _parsers, ticker, default_parser=class_parser.ClassParser())
  name = name if name else str(type(obj))
  selected_parser = state.find_parser(obj)
  if selected_parser is None:
    return
  selected_parser.parse(state, name, obj)
  ticker.register_listener(state.updater.update_all)


_internal = re.compile('__.*__$')


def _non_callable_attrs(obj, attr):
  if _internal.match(attr) is not None:
    return False
  return not callable(getattr(obj, attr))


def all_non_callable_attrs(obj) -> List[str]:
  """Returns the name of all the non callable attributes of obj.

  Use with the multiscope.reflect_attrs decorator to automatically
  create writers for all non-callable attributes of a class.

  Args:
      obj: the object to get the attributes from (often equal to self).

  Returns:
     the list of attributes for Multiscope to parse.
  """
  return [attr for attr in dir(obj) if _non_callable_attrs(obj, attr)]


def register_parser(prsr: parser.Parser):
  """Register a new parser to use when using reflect."""
  _parsers.insert(0, prsr)
