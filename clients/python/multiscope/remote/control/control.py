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

"""Utilities for changing multiscope behaviour conditionally."""

from absl import flags

DISABLE_MULTISCOPE = flags.DEFINE_bool(
    "multiscope_disable",
    False,
    "Globally disable multiscope and cause its calls to become no-ops.",
)


# Module local variable that controls whether all calls to multiscope are
# no-ops. Useful for conditionally disabling multiscope, eg. in tests.
_disabled = False


def undo_disable():
    """If the `multiscope_disable` flag is not set, this re-enables multiscope."""
    global _disabled
    _disabled = False


def disable():
    """All multiscope calls become no-ops."""
    global _disabled
    _disabled = True


def disabled():
    """Returns true if multiscope is disabled."""
    return _disabled or DISABLE_MULTISCOPE.value
