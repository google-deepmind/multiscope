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

"""Write a reference tensor to make sure the display is correct."""

import random
import time

from absl import flags
from numpy import random as nprandom

from absl import app
import multiscope
from multiscope.examples import examples

FLAGS = flags.FLAGS


def main(_):
  multiscope.start_server()
  nested_writer = multiscope.NestedWriter('Nested')
  for _ in examples.step():
    nested = {
        'tensor': nprandom.uniform(low=-1, high=1, size=(20, 20)),
        'children': {
            'positive': nprandom.uniform(low=0, high=1, size=(20, 20)),
            'negative': nprandom.uniform(low=-1, high=0, size=(20, 20)),
        },
        'scalar': random.random(),
        'time': time.strftime('%H:%M:%S %Z on %b %d, %Y'),
    }
    nested_writer.write(nested)


if __name__ == '__main__':
  app.run(main)
