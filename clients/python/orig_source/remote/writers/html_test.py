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
