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

"""Remote multiscope over GRPC."""
from absl import flags

from multiscope.remote.clock import Clock
import multiscope.remote.events
from multiscope.remote.group import Group
from multiscope.remote.server import reset
from multiscope.remote.server import start_server
from multiscope.remote.writers.dataframe import DataFrameWriter
from multiscope.remote.writers.dataspec import DataSpecWriter
from multiscope.remote.writers.html import HTMLWriter
from multiscope.remote.writers.image import ImageWriter
from multiscope.remote.writers.scalar import ScalarWriter
from multiscope.remote.writers.svg import SVGWriter
from multiscope.remote.writers.tensor import TensorWriter
from multiscope.remote.writers.text import TextWriter

flags.DEFINE_bool(
    'multiscope_strict_mode',
    default=False,
    help='Enable multiscope strict mode, which throws exceptions on multiscope-related errors.'
)
