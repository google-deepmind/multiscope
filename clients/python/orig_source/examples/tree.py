"""Example showing how to write a d3-hierarchy tree to Multiscope."""
import json
import time

from absl import app

import multiscope
from multiscope.examples import examples

# See https://github.com/d3/d3-hierarchy#hierarchy for what the format should
# be.
tree = {
    'name':
        'root',
    'id':
        '1',
    'children': [{
        'name': 'hello world',
        'id': '2',
    }, {
        'name': 'step',
        'id': '3',
        'children': [{
            'name': 'empty',
            'id': '4',
        }],
    }],
}


def main(_):
  multiscope.start_server()
  writer = multiscope.TreeWriter('Tree')
  all_children = tree['children']
  for s in examples.step():
    all_children[1]['children'][0]['name'] = str(s)
    # Add or remove nodes every second to exercise the UI.
    if int(time.time()) % 2 == 0:
      tree['children'] = tree['children'][:1]
    else:
      tree['children'] = all_children
    writer.write(json.dumps(tree))


if __name__ == '__main__':
  app.run(main)
