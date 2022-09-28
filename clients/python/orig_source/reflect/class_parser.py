"""Parser for Python classes.

Default parser used when nothing else can match.
"""

from multiscope.reflect import parser
from multiscope.reflect import updater
from multiscope.remote import group

_multiscope_export_attrs = "multiscope_export_attrs"


def reflect_attrs(func):
  """Annotation to mark a callable as exporting attributes for Multiscope.

  Args:
      func: function to mark as a Multiscope attribute exporter.

  Returns:
     the func passed as func.

  Example:

  class MyClass:
    def __init__(self):
      self.my_scalar = 10

    @multiscope.reflect_attrs
    def export_attrs(self):
      return multiscope.all_non_callable_attrs(self)

  will automatically create a scalar writer for the attribute `my_scalar`.
  """
  func.__dict__[_multiscope_export_attrs] = True
  return func


class ClassParser(parser.Parser):
  """Parse a class by parsing all its fields and methods."""

  def can_parse(self, obj):
    """Returns true if this parser can parse `obj`.

    Args:
      obj: An object to build a Multiscope tree for.
    """
    return obj is not None

  def _build_attr_list(self, obj):
    attrs = set()
    for attr_name in dir(obj):
      if attr_name.startswith("_"):
        continue
      attr = getattr(obj, attr_name)
      if hasattr(attr, _multiscope_export_attrs):
        attrs.update(attr())
    return attrs

  def parse(self, state: parser.State, name: str, obj: updater.ReflectTarget):
    """Build a subtree to represent `obj` under the parent node `parent`.

    Args:
       state: Current state of the parser.
       name: The name of the variable to write.
       obj: An instance of the variable to write.
    """
    grp = group.Group(name, state.parent.node)
    state.push_parent(parser.Parent(grp, obj))
    for attr in self._build_attr_list(obj):
      child = getattr(obj, attr)
      if state.has_parent(child):
        continue
      prsr = state.find_parser(child)
      if prsr is None:
        continue
      prsr.parse(state, attr, child)
    state.pop_parent()

  def new_abstract_writer(self, unused_name: str,
                          unused_parent_node: group.ParentNode,
                          unused_obj: updater.ReflectTarget):
    raise NotImplementedError
