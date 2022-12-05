"""A Multiscope Group is directory of Multiscope nodes in the tree."""

import abc
from typing import Optional, Text, Tuple

import six

# from golang.multiscope.streams import writers_pb2 as writers_pb
from multiscope.protos import tree_pb2
from multiscope.protos import base_pb2
from multiscope.protos import base_pb2_grpc
from multiscope.remote import control
from multiscope.remote import stream_client
from multiscope.remote.writers import base


@six.add_metaclass(abc.ABCMeta)
class ParentNode(base.Node):
  """Parent node to which children can be added."""


def join_path(parent: ParentNode, name: str) -> Tuple[str, ...]:
  path = parent.path if parent else ()
  return path + (name,)


def join_path_pb(parent: ParentNode, name: str) -> tree_pb2.NodePath:
  node_path = tree_pb2.NodePath()
  node_path.path.extend(join_path(parent, name))
  return node_path


class Group(ParentNode):
  """Multiscope parent node."""

  @control.init
  def __init__(self, name: str, parent: Optional[ParentNode] = None):
    self._client = base_pb2_grpc.BaseWritersStub(stream_client.channel)
    node_path = join_path_pb(parent, name)
    resp = self._client.NewGroup(base_pb2.NewGroupRequest(path=node_path))
    # self.name = name  # This gets overwritten in the superclass constructor.
    super().__init__(path=tuple(resp.grp.path))
