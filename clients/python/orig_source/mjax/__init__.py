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

"""Multiscope JAX support."""

from absl import flags

from multiscope.mjax import parser
from multiscope.mjax.jit_writer import JITWriter
from multiscope.mjax.jit_writer import write_if_active
from multiscope.reflect import parsers
from multiscope.reflect.nested_writer import NestedWriter
# Import purely for flag definitions.
import multiscope.remote

parsers.register_parser(parser.JAXParser())
