"""Run the Python multiscope instrumentation for ETE tests."""
import math
from typing import Sequence, Text

from absl import app
from absl import flags
import altair as alt
import numpy as np
import pandas as pd

from golang.multiscope.streams.userinputs import userinputs_pb2
from multiscope import remote as multiscope

_HTTP_PORT = flags.DEFINE_integer('http_port', None,
                                  'HTTP port for the multiscope web server')


class Fixture(object):
  """Fixture for ETE test."""

  def __init__(self):
    self.clock = multiscope.Clock('main')
    self.text_writer = multiscope.TextWriter('TextWriter', self.clock)
    self.scalar_writer = multiscope.ScalarWriter('ScalarWriter', self.clock)
    self.data_frame_writer = multiscope.DataFrameWriter('TableWriter',
                                                        self.clock)
    self.data_frame_writer.chart = alt.Chart(
        alt.Data(values=[]), width=400, height=400).mark_rect().encode(
            x='x:O', y='y:O', color='z:Q')
    self.image_writer = multiscope.ImageWriter('ImageWriter', self.clock)
    self.event_text_writer = multiscope.TextWriter('Events', self.clock)
    multiscope.events.register_keyboard_callback(self.handle_keyboard_event,
                                                 self.event_text_writer.path)
    multiscope.events.register_mouse_callback(self.handle_mouse_event,
                                              self.event_text_writer.path)

  def tick(self):
    self.clock.tick()

  def write_text(self, step: int):
    self.text_writer.write('TextWriter. Clock says %d' % step)

  def write_scalar(self, step: int):
    self.scalar_writer.write({'sin': math.sin(step)})

  def write_data_frame(self):
    x, y = np.meshgrid(range(10), range(10))
    data = np.random.uniform(size=(10, 10))
    self.data_frame_writer.write(
        pd.DataFrame.from_dict({
            'x': x.ravel(),
            'y': y.ravel(),
            'z': data.ravel(),
        }))

  def write_image(self):
    image = np.random.random((100, 100)) * 255
    self.image_writer.write(image.astype('uint8'))

  def handle_keyboard_event(self, _: Sequence[Text],
                            event: userinputs_pb2.KeyboardEvent):
    self.event_text_writer.write(f'keyboard event: {event}')

  def handle_mouse_event(self, _: Sequence[Text],
                         event: userinputs_pb2.MouseEvent):
    self.event_text_writer.write(f'mouse event: {event}')


def main(_):
  multiscope.start_server(_HTTP_PORT.value)
  f = Fixture()
  for i in range(int(1e9)):
    f.write_text(i)
    f.write_scalar(i)
    f.write_data_frame()
    f.write_image()
    f.tick()


if __name__ == '__main__':
  app.run(main)
