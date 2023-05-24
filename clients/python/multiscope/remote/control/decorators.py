# Copyright 2023 Google LLC
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

"""Decorators for controlling writers."""
import functools

from absl import flags
from absl import logging
import pytype_extensions

from multiscope.remote import stream_client
from multiscope.remote.control import control


def _suppress_exception(msg, func, is_method: bool):
    """Log and suppress exceptions unless strict mode enabled."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Log and suppress exceptions unless strict mode enabled."""
        # Some code creates writers before calling app.run, this is probably going
        # to fail because multiscope.start_server hasn't been called yet, but don't
        # crash.
        if flags.FLAGS.is_parsed() and flags.FLAGS.multiscope_strict_mode:
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except Exception:  # pylint: disable=broad-except
            if is_method:
                args[0].enabled = False  # Ensure future calls are silent no-ops.
            logging.warning(msg, exc_info=True)
            return None

    return wrapper


def suppress_exception(msg, is_method: bool = False):
    @pytype_extensions.Decorator
    def decorator(func):
        return _suppress_exception(msg, func, is_method)

    return decorator


@pytype_extensions.Decorator
def init(init_func):
    """Decorator to handle exceptions from writer initialization."""

    @suppress_exception(
        "Failed to instantiate Multiscope writer, calls to this writer will be "
        + "ignored"
    )
    @functools.wraps(init_func)
    def wrapper(self, *args, **kwargs):
        """Mark writer as disabled if exception thrown."""
        self.enabled = False
        self._reset_epoch = (
            stream_client.ResetEpoch()
        )  # pylint: disable=protected-access
        # Early return if all multiscope calls are disabled.
        if control.disabled():
            return
        self.path = ()
        if not stream_client.Initialized():
            raise RuntimeError(
                "Tried to initialize multiscope writer before calling "
                + "multiscope.start_server."
            )
        init_func(self, *args, **kwargs)
        self.enabled = True

    return wrapper


@pytype_extensions.Decorator
def method(method_func):
    """Decorator to handle exceptions on writer methods."""

    @functools.wraps(method_func)
    @suppress_exception(
        "Multiscope writer method threw exception, ignoring",
        is_method=True,
    )
    def wrapper(self, *args, **kwargs):
        """Handle exceptions."""
        if not self.enabled:
            return None
        if (
            self._reset_epoch != stream_client.ResetEpoch()
        ):  # pylint: disable=protected-access
            raise RuntimeError(
                "Cannot use a writer that was instantiated before a call to "
                + "multiscope.reset()"
            )
        return method_func(self, *args, **kwargs)

    return wrapper
