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
