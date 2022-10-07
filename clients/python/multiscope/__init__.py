"""Multiscope."""

from absl import flags

from multiscope.remote.server import start_server
from multiscope.remote.writers.text import TextWriter

flags.DEFINE_bool(
    'multiscope_strict_mode',
    default=True,
    help='Enable multiscope strict mode, which throws exceptions on multiscope-related errors.'
)

