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
"""Maintains the list of active paths in Multiscope."""

import collections
import threading
from typing import Callable, List, Mapping, Tuple

from absl import logging
from multiscope.protos import tree_pb2 as pb


class ActivePath:
  """Dispatch active events to writers."""

  def __init__(self, py_client):
    self._path_to_callables: Mapping[Tuple[str, ...], List[Callable[
        [bool], None]]] = collections.defaultdict(list)
    self._main_lock = threading.Lock()
    self._py_client = py_client

  def _call(self, f: Callable[[bool], None], is_active: bool):
    try:
      f(is_active)
    except Exception:  # pylint: disable=broad-except
      logging.warning("active_paths callback error", exc_info=True)

  def run(self):
    """Run the active path update thread."""
    # Listen to active paths.
    threading.Thread(
        target=self._run_updates,
        name="active_path_thread",
        daemon=True,
    ).start()

  def _run_updates(self):
    """Connect to Multiscope server to start a stream to maintain the list of active paths from client requests.

    This function is the main of demon thread updating the list of active paths
    for Multiscope with respect to how web clients connect to the Mutiscope
    server.
    """
    updates = self._py_client.TreeClient().ActivePaths(
        pb.ActivePathsRequest(tree_id=self._py_client.TreeID()))
    last_active = set()
    for paths in updates:
      current_active = {tuple(pbpath.path) for pbpath in paths.paths}
      with self._main_lock:
        for path in current_active:
          for func in self._path_to_callables.get(path, []):
            self._call(func, True)
        for path in last_active - current_active:
          for func in self._path_to_callables.get(path, []):
            self._call(func, False)
      last_active = current_active

  def register_callback(self, path: Tuple[str, ...], callback: Callable[[bool],
                                                                        None]):
    """Register a callback for a given path."""
    with self._main_lock:
      self._path_to_callables[path].append(callback)
