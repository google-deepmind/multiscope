"""Example of how to use Multiscope with matplotlib."""

import io
import time
from absl import app
import matplotlib.pyplot as plt
import numpy as np

import multiscope
from multiscope.examples import examples


def figure_to_np(fig: "figure.Figure") -> np.ndarray:
  """Convert matplotlib figure to numpy array.

  Args:
    fig: The `matplotlib.figure.Figure` to convert. Must have a canvas attached,
      eg. as a result of creating with `plt.figure()`. The figure will be drawn
      to its canvas as a side effect, but not closed.

  Returns:
    A numpy array with shape [height, width, RGB].
  """
  fig.canvas.draw()  # Draw the canvas, cache the renderer.
  im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  # [height, width, RGB]
  return im.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def figure_to_svg(fig: "figure.Figure") -> str:
  """Convert matplotlib figure to an SVG image.

  Args:
    fig: The `matplotlib.figure.Figure` to convert.

  Returns:
    The XML-based representation of an SVG image, as a string.
  """
  data = io.BytesIO()
  fig.savefig(data, format="svg")
  return data.getvalue().decode("utf-8")


def main(_):
  multiscope.start_server()
  w1 = multiscope.ImageWriter('numpy')
  w2 = multiscope.SVGWriter('svg')

  for t in examples.step():
    # Pay the cost of plotting only when either panel is being observed in the
    # UI.
    if w1.should_write or w2.should_write:
      # Increase the dpi for a higher-res (but slower to render) plot.
      fig = plt.figure(dpi=200)
      x = np.linspace(t - 10, t, 100)
      plt.plot(x, np.sin(x))

    # Pay the cost of rendering only when the panel is being observed in the UI.
    if w1.should_write:
      # Option 1: convert to numpy and write to multiscope. Currently this is
      # better supported and more efficient. Change dpi or figure size to adjust
      # the image produced.
      w1.write(figure_to_np(fig))

    # Pay the cost of rendering only when the panel is being observed in the UI.
    if w2.should_write:
      # Option 2: Convert to SVG and write to multiscope.
      w2.write(figure_to_svg(fig))

    if w1.should_write or w2.should_write:
      # Remember to close the figure.
      plt.close(fig)

    time.sleep(0.5)


if __name__ == '__main__':
  app.run(main)
