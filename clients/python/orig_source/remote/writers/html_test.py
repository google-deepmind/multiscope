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

from multiscope import remote as multiscope
from absl.testing import absltest


def setUpModule():
  multiscope.start_server(0)


class TestHtmlWriter(absltest.TestCase):

  def testWriter(self):
    """Creates an HTML writer and then writes to it."""
    w = multiscope.HTMLWriter('writer')
    w.write('<p>Some html</p>')
    w.writeCSS("""
    p {
      font-family: serif;
    }
    """)


if __name__ == '__main__':
  absltest.main()
