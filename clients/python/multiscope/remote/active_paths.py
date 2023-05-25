"""Maintains the list of active paths in Multiscope."""

import collections
import threading

from typing import Callable, List, Mapping, Tuple

from absl import logging

from multiscope.protos import tree_pb2 as pb
from multiscope.remote import stream_client

path_to_callables: Mapping[Tuple[str, ...],
                           List[Callable[[bool],
                                         None]]] = collections.defaultdict(list)
main_lock = threading.Lock()


def _call(f: Callable[[bool], None], is_active: bool):
    try:
        f(is_active)
    except Exception:  # pylint: disable=broad-except
        logging.warning("active_paths callback error", exc_info=True)


def run_updates():
    """Connect to Multiscope server to start a stream to maintain the list of active paths from client requests.

    This function is the main of demon thread updating the list of active paths
    for Multiscope with respect to how web clients connect to the Mutiscope
    server.
    """
    updates = stream_client.ActivePaths(pb.ActivePathsRequest())
    last_active = set()
    for paths in updates:
        current_active = {tuple(pbpath.path) for pbpath in paths.paths}
        with main_lock:
            for path in current_active:
                for func in path_to_callables.get(path, []):
                    _call(func, True)
            for path in last_active - current_active:
                for func in path_to_callables.get(path, []):
                    _call(func, False)
        last_active = current_active


def register_callback(path: Tuple[str, ...], callback: Callable[[bool], None]):
    """Register a callback for a given path."""
    with main_lock:
        path_to_callables[path].append(callback)
