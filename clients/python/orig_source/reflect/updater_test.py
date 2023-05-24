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

"""Tests the updater in the reflect package."""

from golang.multiscope.streams.tables import table_pb2 as table_pb
from golang.stream import stream_pb2 as pb
import multiscope
from multiscope.remote import stream_client
from absl.testing import absltest


class Data:

  def __init__(self):
    self.scalar_a = 42.0
    self.scalar_b = -42.0
    self.accessed = set()

  def clear_accessed(self):
    self.accessed = set()

  def del_attrs(self):
    delattr(self, 'scalar_a')
    delattr(self, 'scalar_b')
    self.value = 42

  def __getattr__(self, name):
    self.accessed.add(name)
    if name == 'scalar_a':
      return self.value
    if name == 'scalar_b':
      return self.value + 1

  @multiscope.reflect_attrs
  def export_attrs(self):
    return multiscope.all_non_callable_attrs(self)


# TODO: the path in `_pbtype` may be incorrect due to package names / paths  # pylint: disable=g-bad-todo
# changing.
_pbtype = 'type.googleapis.com/golang.multiscope.tables.Table'


def _to_dict(read_data):
  vals = {}
  for data in read_data:
    if data.pb.type_url != _pbtype:
      continue
    tbl = table_pb.Table.FromString(data.pb.value)
    for i in range(tbl.length):
      for col in tbl.columns:
        if col.name == '__time__':
          continue
        if col.name == 'Label':
          label = col.string_data[i]
        if col.name == 'Value':
          val = col.float_data[i]
      vals[label] = val
  return vals


class UpdaterTest(absltest.TestCase):

  def _read_data(self, paths):
    pb_paths = [pb.NodePath(path=path) for path in paths]
    req = pb.NodeDataRequest(paths=pb_paths)
    pb_data = stream_client.GetNodeData(req)
    return _to_dict(pb_data.node_data)

  def test_updater(self):
    multiscope.start_server()
    ticker = multiscope.Ticker('main')
    data = Data()
    multiscope.reflect(ticker, 'data', data)  # pylint: disable=not-callable
    data.del_attrs()

    ticker.tick()
    self.assertEqual({'dtype'}, data.accessed)

    # Check that only scalar_a is accessed.
    read_data = {}
    while 'scalar_a' not in read_data.keys():
      read_data = self._read_data([['main', 'data', 'scalar_a', 'data']])
      ticker.tick()
    self.assertEqual(read_data, {'scalar_a': 42.0})
    self.assertEqual({'scalar_a', 'dtype'}, data.accessed)

    # Check that both scalar_a and scalar_b are accessed.
    data.clear_accessed()
    read_data = {}
    while 'scalar_b' not in read_data.keys():
      read_data = self._read_data([
          ['main', 'data', 'scalar_a', 'data'],
          ['main', 'data', 'scalar_b', 'data'],
      ])
      ticker.tick()
    self.assertEqual(read_data, {
        'scalar_a': 42.0,
        'scalar_b': 43.0,
    })
    self.assertEqual(set(['scalar_a', 'scalar_b']), data.accessed)

    # Check that, at some point, no fields are accessed.
    # This is going to last the time a path is active in the server.
    # (about one minute at the time of writing this code).
    n_tick = 0
    while data.accessed:
      data.clear_accessed()
      ticker.tick()
      n_tick = n_tick + 1
    self.assertGreater(n_tick, 2,
                       'disabling access should not have instantaneous')


if __name__ == '__main__':
  absltest.main()
