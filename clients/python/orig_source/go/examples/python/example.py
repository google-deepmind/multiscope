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
