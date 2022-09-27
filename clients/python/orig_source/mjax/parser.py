"""Parser for JAX DeviceArray."""

from jax import numpy as jnp

from multiscope.reflect import parser
from multiscope.reflect import parsers
from multiscope.reflect import updater
from multiscope.remote import group
from multiscope.remote.writers import scalar
from multiscope.remote.writers import tensor


class JAXParser(parser.Parser):
  """Parse numerical values."""

  def can_parse(self, obj):
    """Returns true if this parser can parse `obj`.

    Args:
      obj: An object to build a Multiscope tree for.
    """
    return isinstance(obj, jnp.DeviceArray)

  def parse(self, state: parser.State, name: str, obj: updater.ReflectTarget):
    """Build a subtree to represent `obj` under the parent node `parent`.

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
                          obj: updater.ReflectTarget, force_write: bool):
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
