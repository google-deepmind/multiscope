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
"""Utilities for controlling writers. Internal."""
from multiscope.remote.control.control import disable
from multiscope.remote.control.control import disabled
from multiscope.remote.control.control import undo_disable
from multiscope.remote.control.control import DISABLE_MULTISCOPE
from multiscope.remote.control.decorators import init
from multiscope.remote.control.decorators import method
from multiscope.remote.control.decorators import suppress_exception
