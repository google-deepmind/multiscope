"""Multiscope."""
import warnings

from multiscope.reflect.class_parser import reflect_attrs
from multiscope.reflect.nested_writer import NestedWriter
from multiscope.reflect.parsers import all_non_callable_attrs
from multiscope.reflect.parsers import reflect
# Import purely for flag definitions.
import multiscope.remote
from multiscope.remote.clock import Clock
from multiscope.remote.control.control import disable
from multiscope.remote.control.control import DISABLE_MULTISCOPE
from multiscope.remote.events import events
from multiscope.remote.group import Group
from multiscope.remote.group import ParentNode
from multiscope.remote.server import get_dashboard_url
from multiscope.remote.server import reset
from multiscope.remote.server import server_address
from multiscope.remote.server import start_server
from multiscope.remote.writers.base import Writer
from multiscope.remote.writers.dataframe import DataFrameWriter
from multiscope.remote.writers.dataspec import DataSpecWriter
from multiscope.remote.writers.html import HTMLWriter
from multiscope.remote.writers.image import ImageWriter
from multiscope.remote.writers.loupe import LoupeWriter
from multiscope.remote.writers.scalar import ScalarWriter
from multiscope.remote.writers.svg import SVGWriter
from multiscope.remote.writers.tensor import TensorWriter
from multiscope.remote.writers.text import TextWriter
from multiscope.remote.writers.tree import TreeWriter
from multiscope.remote.writers.webcam import Webcam

Ticker = Clock

warnings.simplefilter('always', DeprecationWarning)


def register_input_event_callback(unused_event_id, unused_function_path):
  warnings.warn(
      'register_input_event_callback is a deprecated no-op, use ' +
      'multiscope.events instead', DeprecationWarning)
