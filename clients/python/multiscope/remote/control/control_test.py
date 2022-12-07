# TODO: does not work yet!

from absl.testing import absltest

import multiscope
from multiscope.protos import root_pb2 as root_pb
from multiscope.protos import tree_pb2 as pb
from multiscope.remote import stream_client
from multiscope.remote.control import control


class TestControl(absltest.TestCase):
    def setUp(self):
        multiscope.enable()  # In case a test disables it.
        return super().setUp()

    def testDisable(self):
        multiscope.disable()
        self.assertIsNone(multiscope.start_server())
        # Strict mode is enabled in BUILD rule for all tests.
        s = multiscope.ScalarWriter("scalar")
        t = multiscope.TextWriter("text")
        t.write("some text")
        if t.should_write:
            t.write("some more text")
        s.write({})
        control._disabled = False

    def testReset(self):
        text_data = "foo"
        multiscope.start_server()
        t = multiscope.Ticker("ticker")
        w = multiscope.TextWriter("text", t)

        multiscope.reset()

        # TODO: make a more specific error check. These used to be:
        # expected_err = "instantiated before a call to multiscope.reset"
        # with self.assertRaisesRegex(RuntimeError, expected_err):
        #   etc
        # but that probably was under not strict mode. With strict mode we
        # first get another exception from grpc.
        with self.assertRaises(Exception):
            t.tick()
        with self.assertRaises(Exception):
            w.write(text_data)

        t = multiscope.Ticker("ticker")
        w = multiscope.TextWriter("text", t)
        w.write(text_data)
        t.tick()

        # Nodes should not be renamed.
        self.assertEqual(t.path, ("ticker",))
        self.assertEqual(
            w.path,
            (
                "ticker",
                "text",
            ),
        )

        # Make sure there are only 2 nodes present.
        request = pb.NodeDataRequest()
        layout_path = pb.NodePath()
        layout_path.path.append(".layout")
        datarequest = pb.DataRequest(path=layout_path, lastTick=0)
        request.reqs.append(datarequest)
        node_data_pb = stream_client.GetNodeData(request).node_data[0].pb
        # TODO: there is no Root anymore, can't see explicit_paths, etc.
        self.assertTrue(node_data_pb.Is(root_pb.Root.DESCRIPTOR))
        root_data = root_pb.Root()
        node_data_pb.Unpack(root_data)
        self.assertCountEqual(
            [t.path, w.path],
            [tuple(p.path) for p in root_data.explicit_paths],
            "layout did not contain expected nodes",
        )

        # We can read the data written.
        request = pb.NodeDataRequest()
        writer_path = pb.NodePath()
        writer_path.path.extend(w.path)
        request.paths.append(writer_path)
        got_text = stream_client.GetNodeData(request).node_data[0].raw.decode("utf-8")
        self.assertEqual(got_text, text_data)


if __name__ == "__main__":
    absltest.main()
