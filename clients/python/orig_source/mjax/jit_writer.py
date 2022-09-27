"""Helper functions to use jax.id_tap with Multiscope."""

import collections
import functools
import threading
from typing import Dict, NamedTuple

from jax import lax
from jax.experimental import host_callback as hcb
import numpy as np

from multiscope.remote import group
from multiscope.remote.writers import tensor

_id_to_jitwriter = []
_jitwriter_lock = threading.Lock()


class _JITWriterData(NamedTuple):
  # ID to which this handler belongs to.
  jitwriter_id: int
  # State of activation of the different writers.
  actives: Dict[str, bool]


class JITWriter:
  """Writes tensors from JAX jitted functions."""

  def __init__(self, name: str):
    self._lock = threading.Lock()
    self.self_id = self._new_static_id()
    self._name_to_writer = collections.defaultdict(lambda: None)
    self._actives = {}
    self.parent = None if name is None else group.Group(name)

  def write_array(self, name: str, array: np.ndarray):
    """Write a JAX tensor given a name."""
    # write_array is called under two conditions:
    # 1. lax.cond selects the branch to write the data to the writer
    # 2. the writer has not been allocated yet. Consequently:
    #    * self._name_to_id will return a 0 ID by default,
    #    * self._actives[0] is always True,
    #    * lax.cond will select the active branch,
    #    * write_array needs to create the writer, so that it is
    #      available to the user (but no data needs to be written
    #      to that writer). In doing so, the writer will get a
    #      non-zero ID and self._actives will be reallocated.
    with self._lock:
      writer = self._name_to_writer[name]
      if writer is None:
        writer = tensor.TensorWriter(name, parent=self.parent)
        self._name_to_writer[name] = writer
        # Disable the writer by default.
        self._actives[name] = False
        # We do not write the tensor to prevent Multiscope
        # from having to allocate memory for all possible tensors
        # that are available.
        return
      writer.write(array)

  def handle(self) -> _JITWriterData:
    with self._lock:
      # TODO: update actives asynchronously, reusing on-device  # pylint: disable=g-bad-todo
      #  tensor if possible.
      self._update_actives_unsafe()
      return _JITWriterData(self.self_id, self._actives)

  def _update_actives_unsafe(self):
    for name, writer in self._name_to_writer.items():
      self._actives[name] = writer.should_write

  def _new_static_id(self) -> int:
    with _jitwriter_lock:
      self_id = len(_id_to_jitwriter)
      _id_to_jitwriter.append(self)
      return self_id


def _id_tap_write(args, transforms, *, name):
  """Write a tensor to Multiscope."""
  del transforms

  jitwriter_id, data = args

  # Tracing.
  if jitwriter_id.dtype == bool:
    return

  with _jitwriter_lock:
    jitwriter = _id_to_jitwriter[jitwriter_id]
  jitwriter.write_array(name, data)


def _id_tap(handle: _JITWriterData, name: str, array: np.ndarray):
  writer = functools.partial(_id_tap_write, name=name)
  return hcb.id_tap(writer, (handle.jitwriter_id, array), result=array)


def write_if_active(handle: _JITWriterData, name: str,
                    array: np.ndarray) -> np.ndarray:
  """Implements the identity function, possibly writing to multiscope as a side-effect.

  Moves data from the device to the host to write it to multiscope, only if the
  corresponding writer is active (e.g. being observed in the multiscope UI).
  Does nothing otherwise.

  Args:
    handle: returned from JITWriter.handle()
    name: name of the tensor, must be unique across this JITWriter
    array: tensor to visualise

  Returns:
    `array` unmodified
  """
  return lax.cond(
      handle.actives is not None and handle.actives.get(name, True),
      lambda _: _id_tap(handle, name, array), lambda _: array, None)
