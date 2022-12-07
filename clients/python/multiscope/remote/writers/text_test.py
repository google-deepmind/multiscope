from absl import flags

import multiscope
from multiscope.remote.writers import text
from absl.testing import absltest


FLAGS = flags.FLAGS
FLAGS.multiscope_strict_mode = True


def setUpModule():
    # TODO: in scalar_test `start_server()` is imported from `remote.__init__`,
    #   here it's from multiscope. Make consistent.
    multiscope.start_server()


class TestTextWriter(absltest.TestCase):
    def testWriter(self):
        """Creates an TextWriter and then writes to it."""
        w = text.TextWriter("writer")
        w.write("Test! Hello World.")


if __name__ == "__main__":
    absltest.main()
