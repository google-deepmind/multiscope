# TODO: does not work yet!

from absl.testing import absltest

import multiscope
from multiscope.protos import root_pb2 as root_pb
from multiscope.protos import tree_pb2 as pb
from multiscope.remote import stream_client
from multiscope.remote.control import control


def _list_all_nodes():
    """Lists all nodes (including the root) in the tree. Not optimized."""
    req = pb.NodeStructRequest()
    req.paths.append(pb.NodePath())
    nodes = stream_client.GetNodeStruct(req).nodes
    if nodes:
        # This needs to be the root node, that's what we requested.
        if len(nodes) != 1 or len(nodes[0].path.path) != 0:
            raise RuntimeError(
                f"We requested the node structure from the root node, but the root "
                f"of the returned tree has {len(nodes)} elements and the first node "
                f"has a non-empty path: {nodes[0].path.path}."
            )
    else:
        return []

    all_nodes = [nodes[0]]
    all_nodes.extend(_list_subtree(nodes[0]))
    return all_nodes


def _list_subtree(root):
    """Returns all nodes in the subtree below the root, not including root."""
    if not root.has_children:
        return []

    if len(root.children) == 0:
        # Request them.
        req = pb.NodeStructRequest()
        req.paths.append(root.path)
        nodes = stream_client.GetNodeStruct(req).nodes
        assert nodes[0].path == root.path
        root = nodes[0]

    all_nodes = [n for n in root.children]
    for n in root.children:
        all_nodes.extend(_list_subtree(n))
    return all_nodes


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

        # Make sure there are only 2 nodes + the root present on the server.
        all_nodes = _list_all_nodes()
        self.assertLen(all_nodes, len(["root", t, w]))

        # We can read the data written.
        node_data_request = pb.NodeDataRequest()
        data_req = pb.DataRequest(lastTick=0)
        data_req.path.path.extend(w.path)
        node_data_request.reqs.append(data_req)
        got_text = (
            stream_client.GetNodeData(node_data_request)
            .node_data[0]
            .raw.decode("utf-8")
        )
        self.assertEqual(got_text, text_data)


if __name__ == "__main__":
    absltest.main()
