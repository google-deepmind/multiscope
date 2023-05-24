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

"""Example testing running Go Multiscope from Python."""

from absl import flags

from absl import app
from multiscope.examples import examples
from multiscope.go.examples.python import gomain_py as gomain

_PORT = flags.DEFINE_integer('port', default=5972, help='Server port.')


def main(_):
  gomain.Start(_PORT.value)
  for t in examples.step():
    gomain.Step(t)


if __name__ == '__main__':
  app.run(main)
