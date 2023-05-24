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

"""Implement the updater called by tickers.

The updater goes over the list of active paths and
calls a `write` callbacks register at these paths.
"""

import abc
import threading
from typing import Any, Callable, Set, NewType

import six


# Name of the type on which we use reflect and then pull data from.
# This type is defined mostly for readability.
ReflectTarget = NewType('ReflectTarget', Any)


@six.add_metaclass(abc.ABCMeta)
class DataPuller:

  @abc.abstractmethod
  def reset(self):
    """Reset the data in the writer."""

  @abc.abstractmethod
  def pull_write(self):
    """Pull data from an instance, then writes it to a writer."""


class ObjectDataPuller(DataPuller):
  """Pull a named attribute to a writer."""

  def __init__(self, writer, name: str, obj: ReflectTarget):
    self.writer = writer
    self.name = name
    self.obj = obj

  def pull_write(self):
    self.writer.write(getattr(self.obj, self.name))

  def reset(self):
    self.writer.reset()


class Updater:
  """Update active writers."""

  def __init__(self):
    self.pullers: Set[DataPuller] = set()
    self.pullers_to_disable: Set[DataPuller] = set()
    self.lock = threading.Lock()

  def enable(self, puller: DataPuller):
    with self.lock:
      self.pullers.add(puller)
      self.pullers_to_disable.discard(puller)

  def disable(self, puller: DataPuller):
    with self.lock:
      self.pullers.discard(puller)
      self.pullers_to_disable.add(puller)

  def new_callback(self, puller: DataPuller) -> Callable[[bool], None]:

    def is_active_callback(is_active: bool):
      if is_active:
        self.enable(puller)
      else:
        self.disable(puller)

    return is_active_callback

  def update_all(self):
    """Updates all active writers.

    Also reset the writers that have transionned from active to non-active.
    """
    with self.lock:
      # TODO: we should able to call them all asynchronously which is  # pylint: disable=g-bad-todo
      # probably worth it because numpy and gRPC calls will be made.
      for puller in self.pullers_to_disable:
        puller.reset()
      for puller in self.pullers:
        puller.pull_write()
      self.pullers_to_disable = set()
