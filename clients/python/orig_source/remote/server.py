"""Multiscope server."""

import threading
from typing import Optional, Text, Tuple

from absl import flags
from absl import logging
import portpicker
import termcolor

from golang.stream import stream_pb2 as pb
from multiscope.cpp.python import server as server_cc
from multiscope.remote import active_paths
from multiscope.remote import control
from multiscope.remote import stream_client

flags.DEFINE_integer(
    'multi_port',
    default=5972,
    help='Port for the multiscope web UI. '
    'Set to 0 to automatically pick an unused port.',
    allow_override=True)

flags.DEFINE_integer(
    'multi_loupe_port',
    default=0,
    help='Port for the multiscope loupe web UI.',
    allow_override=True)

flags.DEFINE_integer(
    'multi_ice_port',
    default=0,
    help='Port for the multiscope loupe webrtc.',
    allow_override=True)

FLAGS = flags.FLAGS
_web_port: Optional[int] = None
_grpc_url: Optional[str] = None
_lock = threading.Lock()


@control.suppress_exception('Failed to start multiscope server')
def start_server(port: Optional[int] = None,
                 experiment_path: Optional[str] = None) -> Optional[int]:
  """Starts the Multiscope server.

  This function may be called multiple times, but the server will only be
  started once. No exceptions will be thrown even if we fail to start unless the
  `multiscope_strict_mode` flag is enabled.

  Args:
    port: requested port for the multiscope web server
    experiment_path: unique identifier used to persist per-experiment settings
      like the dashboard layout. If None, will try to auto-infer the
      experiment path.

  Returns:
     actually picked port for the multiscope web server
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

    port = port or FLAGS.multi_port
    _web_port, _grpc_url = _start_server_unsafe(
        port=port, experiment_path=experiment_path)

  return _web_port


def _start_server_unsafe(port: int,
                         experiment_path: Optional[str]) -> Tuple[int, str]:
  """Starts an http server with Multiscope enabled."""
  if port == 0 or not portpicker.is_port_free(port):
    port = portpicker.pick_unused_port()

  grpc_url = server_cc.StartServer(
      http_port=port,
      loupe_port=FLAGS.multi_loupe_port,
      ice_port=FLAGS.multi_ice_port,
      experiment_path=experiment_path)

  web_url = get_dashboard_url(port)
  termcolor.cprint(
      '[Multiscope] Live dashboard: {}'.format(web_url),
      color='blue',
      attrs=['bold'],
      flush=True,
  )

  stream_client.InitializeStub(grpc_url)
  # Make sure we can connect to the multiscope server.
  stream_client.TryConnecting(timeout_secs=120)

  threading.Thread(
      target=active_paths.run_updates,
      name='active_path_thread',
      daemon=True,
  ).start()

  return port, grpc_url


def get_dashboard_url(port: int) -> Text:
  """Returns the url to connect to Multiscope."""
  return f'http://localhost:{port}'


def server_address() -> Optional[str]:
  """Returns the gRPC address of the multiscope server if it's running."""
  with _lock:
    return _grpc_url


def reset():
  """Resets the multiscope server state by removing all nodes."""
  if control.disabled():
    return
  stream_client.ResetState(pb.ResetStateRequest())
