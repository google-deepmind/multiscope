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

from absl import flags

from multiscope.remote.clock import Ticker
from multiscope.remote.control.control import disable
from multiscope.remote.control.control import undo_disable
from multiscope.remote.control.control import DISABLE_MULTISCOPE
from multiscope.remote.player import Player
from multiscope.remote.server import get_dashboard_url
from multiscope.remote.server import server_address
from multiscope.remote.server import reset
from multiscope.remote.server import start_server
from multiscope.remote.writers.base import Writer
from multiscope.remote.writers.scalar import ScalarWriter
from multiscope.remote.writers.tensor import TensorWriter
from multiscope.remote.writers.text import TextWriter

# Import like this, or explicitly import events from ....events? Right now
# the list of entries is the same.
import multiscope.remote.events

flags.DEFINE_bool(
    "multiscope_strict_mode",
    default=True,
    help="Enable multiscope strict mode, which throws exceptions on multiscope-related errors.",
)
