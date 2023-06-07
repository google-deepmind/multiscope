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
"""Interactions with the server."""

import portpicker
import threading
from typing import Optional, Tuple

import atexit
import logging
import os
import subprocess
import termcolor

from absl import flags

from multiscope.protos import tree_pb2 as pb
from multiscope.remote import control
from multiscope.remote import stream_client

_HTTP_PORT = flags.DEFINE_integer("http_port", 5972, "http port.")
# TODO: grpc_port=0 picks one that works. Good default, but right now we don't
# just pick up the selected port. Temporarily disable it.
_GRPC_PORT = flags.DEFINE_integer("grpc_port", 59160, "grpc port.")
_LOCAL = flags.DEFINE_bool("local", True, "local connections only.")
_MULTISCOPE_BINARY_PATH = flags.DEFINE_string(
    "multiscope_server_path",
    "~/bin/multiscope_server",
    "The path to the pre-built multiscope binary.",
)

_web_port: Optional[int] = None
_grpc_url: Optional[str] = None
_lock = threading.Lock()


def start_server(
    port: Optional[int] = None,
    grpc_port: Optional[int] = None,
    connection_timeout_secs: int = 15,
) -> Optional[int]:
  """Starts the Multiscope server.

    This function may be called multiple times, but the server will only be
    started once. No exceptions will be thrown even if we fail to start unless the
    `multiscope_strict_mode` flag is enabled.

    THE ARGUMENT LIST BELOW IS OUT OF DATE.

    # TODO: update the code below to support the original arguments if possible,
    # and if keeping the `connection_timeout_secs` argument, then put it in
    # the args list.

    Args:
      port: requested port for the multiscope web server
      experiment_path: unique identifier used to persist per-experiment settings
        like the dashboard layout. A good value is the google3 path to the
        experiment target, eg "learning/deepmind/your/experiment/target". If None,
        will try to auto-infer the experiment path, but this doesn't work on borg
        due to the way xmanager packages experiments.

    Returns:
       Actually picked port for the multiscope web server. None if multiscope
       is disabled.
    """
  # Race condition guard.
  with _lock:
    if control.disabled():
      return None

    global _web_port, _grpc_url
    if _web_port is not None:
      logging.warning(
          "multiscope.start_server called more than once; ignoring. "
          "Multiscope dashboard already started at: %s",
          get_dashboard_url(_web_port),
      )
      return _web_port

    http_port = port or _HTTP_PORT.value
    grpc_port = grpc_port or _GRPC_PORT.value

    _web_port, _grpc_url = _start_server_unsafe(
        http_port=http_port,
        grpc_port=grpc_port,
        connection_timeout_secs=connection_timeout_secs,
    )

  return _web_port


def _start_server_unsafe(http_port: int, grpc_port: int,
                         connection_timeout_secs) -> Tuple[int, str]:
  """Starts the server and returns the http and grpc ports."""
  # TODO: this currently requires:
  #
  # 1. That the multiscope is (built manually and) avaiable at
  #    _MULTISCOPE_BINARY_PATH. From the root multiscope/server$ directory, run
  #     go build -o ~/bin/multiscope_server multiscope.go
  #     Solutions:
  #     a) Call into Go from Python:
  #         * Go --> C/C++ (through ctypes or others) --> Python is a path
  #         * using RPC is suggested.
  #     b) Build the multiscope server (if needed?).
  if http_port == 0 or not portpicker.is_port_free(http_port):
    http_port = portpicker.pick_unused_port()
  if grpc_port == 0 or not portpicker.is_port_free(grpc_port):
    grpc_port = portpicker.pick_unused_port()

  server_process = subprocess.Popen([
      os.path.expanduser(_MULTISCOPE_BINARY_PATH.value),
      "--http_port=" + str(http_port),
      "--grpc_port=" + str(grpc_port),
      "--local=" + str(_LOCAL.value),
  ])

  def close_server():
    server_process.terminate()

  atexit.register(close_server)

  web_url = get_dashboard_url(http_port)
  termcolor.cprint(
      "[Multiscope] Live dashboard: {}".format(web_url),
      color="blue",
      attrs=["bold"],
      flush=True,
  )
  grpc_url = f"localhost:{grpc_port}"

  logging.info("Connecting grp_url: %s", grpc_url)
  stream_client.InitGlobalConnection(grpc_url)
  client = stream_client.Client(timeout_secs=connection_timeout_secs)
  stream_client.SetGlobalClient(client)
  return http_port, grpc_url


def server_address() -> str:
  """Returns the Multiscope gRPC servrer url to connect a Multiscope client."""
  return _grpc_url


def get_dashboard_url(port: int) -> str:
  if _LOCAL.value:
    return f"http://localhost:{port}"
  else:
    # TODO: can we figure the host name out?
    return f"<your_host_name>:{port}"


def reset():
  """Resets the multiscope server state by removing all nodes."""
  if control.disabled():
    return
  stream_client.ResetState(pb.ResetStateRequest())
