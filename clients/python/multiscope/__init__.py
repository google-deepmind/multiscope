# Copyright 2023 DeepMind Technologies Limited
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
"""Multiscope. Public facing interface."""

from typing import Callable, Optional

from absl import flags
from multiscope.protos import events_pb2
from multiscope.remote import group
from multiscope.remote import player
from multiscope.remote import stream_client
from multiscope.remote import ticker
from multiscope.remote import userinputs
from multiscope.remote.control.control import disable
from multiscope.remote.control.control import DISABLE_MULTISCOPE
from multiscope.remote.control.control import undo_disable
from multiscope.remote.server import get_dashboard_url
from multiscope.remote.server import reset
from multiscope.remote.server import server_address
from multiscope.remote.server import start_server
from multiscope.remote.writers import base
from multiscope.remote.writers import image
from multiscope.remote.writers import scalar
from multiscope.remote.writers import tensor
from multiscope.remote.writers import text

flags.DEFINE_bool(
    "multiscope_strict_mode",
    default=True,
    help=("Enable multiscope strict mode, which throws exceptions on"
          " multiscope-related errors."),
)

Writer = base.Writer

KeyboardEvent = events_pb2.Keyboard
MouseEvent = events_pb2.Mouse


def global_client():
  return stream_client.GlobalClient()


def ImageWriter(name: str, parent: Optional[group.ParentNode] = None):
  return image.ImageWriter(
      py_client=stream_client.GlobalClient(), name=name, parent=parent)


def Player(name: str,
           parent: Optional[group.ParentNode] = None,
           stoppable: bool = True):
  return player.Player(
      py_client=stream_client.GlobalClient(),
      name=name,
      parent=parent,
      stoppable=stoppable,
  )


def ScalarWriter(name: str, parent: Optional[group.ParentNode] = None):
  return scalar.ScalarWriter(
      py_client=stream_client.GlobalClient(), name=name, parent=parent)


def TensorWriter(name: str, parent: Optional[group.ParentNode] = None):
  return tensor.TensorWriter(
      py_client=stream_client.GlobalClient(), name=name, parent=parent)


def TextWriter(name: str, parent: Optional[group.ParentNode] = None):
  return text.TextWriter(
      py_client=stream_client.GlobalClient(), name=name, parent=parent)


def HTMLWriter(name: str, parent: Optional[group.ParentNode] = None):
  return text.HTMLWriter(
      py_client=stream_client.GlobalClient(), name=name, parent=parent)


def Ticker(name: str, parent: Optional[group.ParentNode] = None):
  return ticker.Ticker(
      py_client=stream_client.GlobalClient(), name=name, parent=parent)


def register_keyboard_callback(cb: Callable[[events_pb2.Mouse], None]):
  return userinputs.register_keyboard_callback(
      py_client=stream_client.GlobalClient(), cb=cb)


def register_mouse_callback(cb: Callable[[events_pb2.Keyboard], None]):
  return userinputs.register_mouse_callback(
      py_client=stream_client.GlobalClient(), cb=cb)
