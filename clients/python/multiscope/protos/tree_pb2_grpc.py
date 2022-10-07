# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from multiscope.protos import tree_pb2 as multiscope_dot_protos_dot_tree__pb2


class TreeStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetNodeStruct = channel.unary_unary(
                '/multiscope.Tree/GetNodeStruct',
                request_serializer=multiscope_dot_protos_dot_tree__pb2.NodeStructRequest.SerializeToString,
                response_deserializer=multiscope_dot_protos_dot_tree__pb2.NodeStructReply.FromString,
                )
        self.GetNodeData = channel.unary_unary(
                '/multiscope.Tree/GetNodeData',
                request_serializer=multiscope_dot_protos_dot_tree__pb2.NodeDataRequest.SerializeToString,
                response_deserializer=multiscope_dot_protos_dot_tree__pb2.NodeDataReply.FromString,
                )
        self.SendEvents = channel.unary_unary(
                '/multiscope.Tree/SendEvents',
                request_serializer=multiscope_dot_protos_dot_tree__pb2.SendEventsRequest.SerializeToString,
                response_deserializer=multiscope_dot_protos_dot_tree__pb2.SendEventsReply.FromString,
                )
        self.ResetState = channel.unary_unary(
                '/multiscope.Tree/ResetState',
                request_serializer=multiscope_dot_protos_dot_tree__pb2.ResetStateRequest.SerializeToString,
                response_deserializer=multiscope_dot_protos_dot_tree__pb2.ResetStateReply.FromString,
                )
        self.ActivePaths = channel.unary_stream(
                '/multiscope.Tree/ActivePaths',
                request_serializer=multiscope_dot_protos_dot_tree__pb2.ActivePathsRequest.SerializeToString,
                response_deserializer=multiscope_dot_protos_dot_tree__pb2.ActivePathsReply.FromString,
                )
        self.StreamEvents = channel.unary_stream(
                '/multiscope.Tree/StreamEvents',
                request_serializer=multiscope_dot_protos_dot_tree__pb2.StreamEventsRequest.SerializeToString,
                response_deserializer=multiscope_dot_protos_dot_tree__pb2.Event.FromString,
                )


class TreeServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetNodeStruct(self, request, context):
        """Browse the structure of the graph.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetNodeData(self, request, context):
        """Request data from nodes in the graph.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendEvents(self, request, context):
        """Send events to the backend.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ResetState(self, request, context):
        """Reset the state of the server including the full tree as well as the events
        registry.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ActivePaths(self, request, context):
        """Returns the list of paths for which the data needs to be written if
        possible.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamEvents(self, request, context):
        """Request a stream of events from the backend.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TreeServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetNodeStruct': grpc.unary_unary_rpc_method_handler(
                    servicer.GetNodeStruct,
                    request_deserializer=multiscope_dot_protos_dot_tree__pb2.NodeStructRequest.FromString,
                    response_serializer=multiscope_dot_protos_dot_tree__pb2.NodeStructReply.SerializeToString,
            ),
            'GetNodeData': grpc.unary_unary_rpc_method_handler(
                    servicer.GetNodeData,
                    request_deserializer=multiscope_dot_protos_dot_tree__pb2.NodeDataRequest.FromString,
                    response_serializer=multiscope_dot_protos_dot_tree__pb2.NodeDataReply.SerializeToString,
            ),
            'SendEvents': grpc.unary_unary_rpc_method_handler(
                    servicer.SendEvents,
                    request_deserializer=multiscope_dot_protos_dot_tree__pb2.SendEventsRequest.FromString,
                    response_serializer=multiscope_dot_protos_dot_tree__pb2.SendEventsReply.SerializeToString,
            ),
            'ResetState': grpc.unary_unary_rpc_method_handler(
                    servicer.ResetState,
                    request_deserializer=multiscope_dot_protos_dot_tree__pb2.ResetStateRequest.FromString,
                    response_serializer=multiscope_dot_protos_dot_tree__pb2.ResetStateReply.SerializeToString,
            ),
            'ActivePaths': grpc.unary_stream_rpc_method_handler(
                    servicer.ActivePaths,
                    request_deserializer=multiscope_dot_protos_dot_tree__pb2.ActivePathsRequest.FromString,
                    response_serializer=multiscope_dot_protos_dot_tree__pb2.ActivePathsReply.SerializeToString,
            ),
            'StreamEvents': grpc.unary_stream_rpc_method_handler(
                    servicer.StreamEvents,
                    request_deserializer=multiscope_dot_protos_dot_tree__pb2.StreamEventsRequest.FromString,
                    response_serializer=multiscope_dot_protos_dot_tree__pb2.Event.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'multiscope.Tree', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Tree(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetNodeStruct(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/multiscope.Tree/GetNodeStruct',
            multiscope_dot_protos_dot_tree__pb2.NodeStructRequest.SerializeToString,
            multiscope_dot_protos_dot_tree__pb2.NodeStructReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetNodeData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/multiscope.Tree/GetNodeData',
            multiscope_dot_protos_dot_tree__pb2.NodeDataRequest.SerializeToString,
            multiscope_dot_protos_dot_tree__pb2.NodeDataReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendEvents(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/multiscope.Tree/SendEvents',
            multiscope_dot_protos_dot_tree__pb2.SendEventsRequest.SerializeToString,
            multiscope_dot_protos_dot_tree__pb2.SendEventsReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ResetState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/multiscope.Tree/ResetState',
            multiscope_dot_protos_dot_tree__pb2.ResetStateRequest.SerializeToString,
            multiscope_dot_protos_dot_tree__pb2.ResetStateReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ActivePaths(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/multiscope.Tree/ActivePaths',
            multiscope_dot_protos_dot_tree__pb2.ActivePathsRequest.SerializeToString,
            multiscope_dot_protos_dot_tree__pb2.ActivePathsReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StreamEvents(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/multiscope.Tree/StreamEvents',
            multiscope_dot_protos_dot_tree__pb2.StreamEventsRequest.SerializeToString,
            multiscope_dot_protos_dot_tree__pb2.Event.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)