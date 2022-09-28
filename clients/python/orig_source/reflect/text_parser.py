"""Parser for text types (str)."""
import numpy as np

from multiscope.reflect import parser
from multiscope.reflect import parsers
from multiscope.reflect import updater
from multiscope.remote import group
from multiscope.remote.writers import text


class TextParser(parser.Parser):
  """Parse text values."""

  def can_parse(self, obj: updater.ReflectTarget):
    """Returns true if this parser can parse `obj`.

    Args:
      obj: An object to build a Multiscope tree for.
    """
    return isinstance(obj, str) or (hasattr(obj, 'dtype') and
                                    obj.dtype == np.dtype('O'))

  def parse(self, state: parser.State, name: str, obj: updater.ReflectTarget):
    """Build a subtree to represent `obj` under the parent node `parent`.

    This method is not used and  required for parsing text so returns None.

    Args:
       state: Current state of the parser.
       name: The name of the variable to write.
       obj: An instance of the variable to write.
    """
    return None

  def new_abstract_writer(self, name: str, parent_node: group.ParentNode,
                          obj: updater.ReflectTarget, force_write: bool):
    wrt = text.TextWriter(name, parent_node)
    return parsers.ConcreteWriter(wrt, str, force_write)
