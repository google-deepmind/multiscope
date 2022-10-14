"""Interactions with the server."""
# TODO: this is a minimal implementation, far from the original.

import threading
from typing import Optional, Tuple

import atexit
import logging
import os
import subprocess

from absl import flags

from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import stream_client


_HTTP_PORT = flags.DEFINE_integer("http_port", 5972, "http port.")
# TODO: grpc_port=0 picks one that works. Good default, but right now we don't
# just pick up the selected port. Temporarily disable it.
_GRPC_PORT = flags.DEFINE_integer("grpc_port", 59160, "grpc port.")
_LOCAL = flags.DEFINE_bool("local", True, "local connections only.")
_MULTISCOPE_BINARY_PATH = flags.DEFINE_string(
  "multiscope_server_path", "~/bin/multiscope_server",
  "The path to the pre-built multiscope binary.")


_web_port: Optional[int] = None
_grpc_url: Optional[str] = None
_lock = threading.Lock()


def start_server(connection_timeout_secs: int = 15) -> Optional[int]:
  """Starts the Multiscope server.

  This function may be called multiple times, but the server will only be
  started once. No exceptions will be thrown even if we fail to start unless the
  `multiscope_strict_mode` flag is enabled.

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
          'multiscope.start_server called more than once; ignoring. '
          'Multiscope dashboard already started at: %s',
          get_dashboard_url(_web_port))
      return _web_port

    _web_port, _grpc_url = start_server_unsafe(connection_timeout_secs)
  return _web_port

def start_server_unsafe(connection_timeout_secs) -> Tuple[int, str]:
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
  # 2. For us to pick a grpc port on the flag that's going to be available.
  #    This is because we don't get, from the server, what grpc_url was chosen;
  #    We may:
  #    a) parse the output from the multiscope server binary we just started.
  #    b) if calling into Go, just fetch the return value.
  #    c) write the url into an agreed upon spot.
  server_process = subprocess.Popen([
      os.path.expanduser(_MULTISCOPE_BINARY_PATH.value),
      "--http_port", str(_HTTP_PORT.value),
      "--grpc_port", str(_GRPC_PORT.value),
      "--local", str(_LOCAL.value),
  ])

  def close_server():
    server_process.terminate()

  atexit.register(close_server)

  if _LOCAL.value:
    grpc_url = f"localhost:{_GRPC_PORT.value}"
  else:
    # TODO: implement.
    raise NotImplementedError("Figure out the grpc_url.")

  logging.info("Connecting grp_url: %s", grpc_url)
  stream_client.InitializeStub(grpc_url)
  # Make sure we can connect to the multiscope server.
  stream_client.TryConnecting(timeout_secs=connection_timeout_secs)

  # TODO: should this be stopped gracefully on exit, or is it fine?
  threading.Thread(
      target=active_paths.run_updates,
      name='active_path_thread',
      daemon=True,
  ).start()

  return _HTTP_PORT.value, grpc_url

def get_dashboard_url(port: int) -> str:
  if _LOCAL.value:
    return f"http://localhost:{port}"
  else:
    # TODO: implement.
    raise NotImplementedError("Figure the base out?")
