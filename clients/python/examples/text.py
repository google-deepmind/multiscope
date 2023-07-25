# Copyright 2023 DeepMind Technologies Limited
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
"""A simple example of using a TextWriter."""

import time

from absl import app
from examples import common
import multiscope


def main(argv):
  multiscope.start_server()
  text = multiscope.TextWriter("Current time Raw text")
  html = multiscope.HTMLWriter("Current time HTML")
  html.write_css("""
.fancy {color: red;}
.superfancy {color: blue;}
""")
  for _ in common.step():
    text.write(time.strftime("%H:%M:%S %Z on %b %d, %Y"))
    html.write(
        '<h1 class="fancy">{time}</h1> on <h1 class="superfancy">{date}</h1>'
        .format(
            time=time.strftime("%H:%M:%S %Z"),
            date=time.strftime("%b %d, %Y"),
        ))


if __name__ == "__main__":
  app.run(main)
