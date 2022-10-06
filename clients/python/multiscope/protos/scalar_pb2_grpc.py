# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import scalar_pb2 as scalar__pb2


class ScalarsStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.NewWriter = channel.unary_unary(
                '/multiscope.scalar.Scalars/NewWriter',
                request_serializer=scalar__pb2.NewWriterRequest.SerializeToString,
                response_deserializer=scalar__pb2.NewWriterResponse.FromString,
                )
        self.Write = channel.unary_unary(
                '/multiscope.scalar.Scalars/Write',
                request_serializer=scalar__pb2.WriteRequest.SerializeToString,
                response_deserializer=scalar__pb2.WriteResponse.FromString,
                )


class ScalarsServicer(object):
    """Missing associated documentation comment in .proto file."""

    def NewWriter(self, request, context):
        """Create a new scalars writer node in Multiscope.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Write(self, request, context):
        """Write scalars to Multiscope.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ScalarsServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'NewWriter': grpc.unary_unary_rpc_method_handler(
                    servicer.NewWriter,
                    request_deserializer=scalar__pb2.NewWriterRequest.FromString,
                    response_serializer=scalar__pb2.NewWriterResponse.SerializeToString,
            ),
            'Write': grpc.unary_unary_rpc_method_handler(
                    servicer.Write,
                    request_deserializer=scalar__pb2.WriteRequest.FromString,
                    response_serializer=scalar__pb2.WriteResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'multiscope.scalar.Scalars', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Scalars(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def NewWriter(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/multiscope.scalar.Scalars/NewWriter',
            scalar__pb2.NewWriterRequest.SerializeToString,
            scalar__pb2.NewWriterResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Write(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/multiscope.scalar.Scalars/Write',
            scalar__pb2.WriteRequest.SerializeToString,
            scalar__pb2.WriteResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
