"""Parser for nested types (Mapping, tuple, list)."""

from typing import Mapping, Sequence

from multiscope.reflect import nested_writer
from multiscope.reflect import parser
from multiscope.reflect import parsers
from multiscope.reflect import updater
from multiscope.remote import group


class NestedParser(parser.Parser):
  """Parse nested values."""

  def can_parse(self, obj: updater.ReflectTarget):
    """Returns true if this parser can parse `obj`.

    Args:
      obj: An object to build a Multiscope tree for.
    """
    return isinstance(obj, Mapping) or isinstance(obj, Sequence)

  def parse(self, state: parser.State, name: str, obj: updater.ReflectTarget):
    """Build a subtree to represent `obj` under the parent node `parent`.

    Args:
       state: Current state of the parser.
       name: The name of the variable to write.
       obj: An instance of the variable to write.
    """
    return None

  def new_abstract_writer(self, name: str, parent_node: group.ParentNode,
                          obj: updater.ReflectTarget, force_write: bool):
    wrt = nested_writer.NestedWriter(name, parent_node, force_write=force_write)
    return parsers.ConcreteWriter(wrt, lambda x: x, force_write)
