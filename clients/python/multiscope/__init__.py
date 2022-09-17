"""Multiscope."""

import atexit
import os
import subprocess

from absl import flags
from typing import Tuple


_HTTP_PORT = flags.DEFINE_integer("http_port", 5972, "http port.")
_GRPC_PORT = flags.DEFINE_integer("grpc_port", 0, "grpc port.")
_LOCAL = flags.DEFINE_bool("local", True, "local connections only.")
_MULTISCOPE_BINARY_PATH = flags.DEFINE_string(
  "multiscope_server_path", "~/bin/multiscope_server",
  "The path to the pre-built multiscope binary.")


def start_server() -> Tuple[int, int]:
  """Starts the server and returns the http and grpc ports."""
  # TODO: add safety?
  return start_server_unsafe()


def start_server_unsafe() -> Tuple[int, int]:
  """Starts the server and returns the http and grpc ports."""
  server = subprocess.Popen([
      os.path.expanduser(_MULTISCOPE_BINARY_PATH.value),
      "--http_port", str(_HTTP_PORT.value),
      "--grpc_port", str(_GRPC_PORT.value),
      "--local", str(_LOCAL.value),
  ])

  def close_server():
    server.terminate()

  atexit.register(close_server)

  return _HTTP_PORT.value, _GRPC_PORT.value


class TextWriter:
  """No-op test implementation."""

  def __init__(self, txt: str):
    pass

  def write(self, stuff):
    pass