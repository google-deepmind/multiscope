"""A Multiscope Group is directory of Multiscope nodes in the tree."""

import abc
from typing import Optional, Text, Tuple

import six

# from golang.multiscope.streams import writers_pb2 as writers_pb
from multiscope.protos import tree_pb2 as pb
from multiscope.remote import control
from multiscope.remote import stream_client
from multiscope.remote.writers import base


@six.add_metaclass(abc.ABCMeta)
class ParentNode(base.Node):
  """Parent node to which children can be added."""


def join_path(parent: ParentNode, name: str) -> Tuple[str]:
  path = parent.path if parent else ()
  # TODO: sort out these types.
  return path + (name,)  # pytype: disable=bad-return-type


def join_path_pb(parent: ParentNode, name: str) -> pb.NodePath:
  node_path = pb.NodePath()
  node_path.path.extend(join_path(parent, name))
  return node_path


# TODO: re-enable once needed.
# class Group(ParentNode):
#   """Multiscope parent node."""

#   @control.init
#   def __init__(self, name: Text, parent: Optional[ParentNode] = None):
#     path = join_path(parent, name)
#     request = pb.CreateNodeRequest()
#     request.path.path.extend(path)
#     request.type = writers_pb.Writers().group
#     resp = stream_client.CreateNode(request=request)
#     self.name = name
#     super().__init__(path=tuple(resp.path.path))
