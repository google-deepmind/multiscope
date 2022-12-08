import time

from absl.testing import absltest

import multiscope
from multiscope import test_control


class TestServer(absltest.TestCase):

    # Users may be starting multiple servers. Although this is a programming
    # error, multiscope should work normally.
    @absltest.skipIf(not test_control.only_fast, "Only running fast tests.")
    def testStartServerMultipleTimes(self):
        for _ in range(10):
            multiscope.start_server()
        # start_server is asynchronous, wait for multiscope to attempt startup
        time.sleep(10)


if __name__ == "__main__":
    absltest.main()
