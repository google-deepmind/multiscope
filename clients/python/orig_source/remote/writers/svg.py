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

"""An SVG writer."""
import base64
import random
import string
from typing import Optional

from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote.writers import base
from multiscope.remote.writers import html


class SVGWriter(base.Writer):
  """Writes an SVG image to multiscope."""
  # TODO: support SVG properly in the frontend. This class abuses  # pylint: disable=g-bad-todo
  #  HTMLWriter to get the job done.

  @control.init
  def __init__(self, name: str, parent: Optional[group.ParentNode] = None):

    self._writer = html.HTMLWriter(name, parent)
    super().__init__(path=self._writer.path)
    # CSS styling applies globally so generate random class.
    self._css_class = ''.join(random.choices(string.ascii_letters, k=10))
    self._writer.write(f'<div class="{self._css_class}"></div>')

  @control.method
  def write(self, svg: str):
    """Write an SVG image to multiscope.

    Args:
      svg: The XML-based representation of the SVG image as a string.
    """
    b64 = base64.b64encode(bytes(svg, 'utf-8')).decode('utf-8')
    # TODO: support scaling properly  # pylint: disable=g-bad-todo
    self._writer.writeCSS(f'''div.{self._css_class} {{
          background-size: 1000px 1000px;
          background-repeat: no-repeat;
          background-image: url(\'data:image/svg+xml;base64,{b64}\');
        }}''')
