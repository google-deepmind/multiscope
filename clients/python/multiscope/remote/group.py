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

"""A Multiscope Group is directory of Multiscope nodes in the tree."""

import abc
from typing import Optional, Tuple

import six

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
    return tree_pb2.NodePath(path=join_path(parent, name))


class Group(ParentNode):
    """Multiscope parent node."""

    @control.init
    def __init__(self, name: str, parent: Optional[ParentNode] = None):
        self._client = base_pb2_grpc.BaseWritersStub(stream_client.channel)
        node_path = join_path_pb(parent, name)
        resp = self._client.NewGroup(base_pb2.NewGroupRequest(path=node_path))
        # TODO: figure out whether we can remove this:
        # self.name = name  # This gets overwritten in the superclass constructor.
        super().__init__(path=tuple(resp.grp.path))
