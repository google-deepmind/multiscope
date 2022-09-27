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
