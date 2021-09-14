# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import libreasr.api.interfaces.libreasr_pb2 as libreasr__pb2


class LibreASRStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Transcribe = channel.unary_unary(
            "/LibreASR.LibreASR/Transcribe",
            request_serializer=libreasr__pb2.AudioAndSettings.SerializeToString,
            response_deserializer=libreasr__pb2.Transcript.FromString,
        )
        self.TranscribeStream = channel.stream_stream(
            "/LibreASR.LibreASR/TranscribeStream",
            request_serializer=libreasr__pb2.AudioOrSettings.SerializeToString,
            response_deserializer=libreasr__pb2.Event.FromString,
        )
        self.PreloadVoice = channel.unary_unary(
            "/LibreASR.LibreASR/PreloadVoice",
            request_serializer=libreasr__pb2.AudioAndSettings.SerializeToString,
            response_deserializer=libreasr__pb2.VoiceClip.FromString,
        )
        self.Translate = channel.unary_unary(
            "/LibreASR.LibreASR/Translate",
            request_serializer=libreasr__pb2.Text.SerializeToString,
            response_deserializer=libreasr__pb2.Text.FromString,
        )


class LibreASRServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Transcribe(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def TranscribeStream(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def PreloadVoice(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def Translate(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_LibreASRServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Transcribe": grpc.unary_unary_rpc_method_handler(
            servicer.Transcribe,
            request_deserializer=libreasr__pb2.AudioAndSettings.FromString,
            response_serializer=libreasr__pb2.Transcript.SerializeToString,
        ),
        "TranscribeStream": grpc.stream_stream_rpc_method_handler(
            servicer.TranscribeStream,
            request_deserializer=libreasr__pb2.AudioOrSettings.FromString,
            response_serializer=libreasr__pb2.Event.SerializeToString,
        ),
        "PreloadVoice": grpc.unary_unary_rpc_method_handler(
            servicer.PreloadVoice,
            request_deserializer=libreasr__pb2.AudioAndSettings.FromString,
            response_serializer=libreasr__pb2.VoiceClip.SerializeToString,
        ),
        "Translate": grpc.unary_unary_rpc_method_handler(
            servicer.Translate,
            request_deserializer=libreasr__pb2.Text.FromString,
            response_serializer=libreasr__pb2.Text.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "LibreASR.LibreASR", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class LibreASR(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Transcribe(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/LibreASR.LibreASR/Transcribe",
            libreasr__pb2.AudioAndSettings.SerializeToString,
            libreasr__pb2.Transcript.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def TranscribeStream(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/LibreASR.LibreASR/TranscribeStream",
            libreasr__pb2.AudioOrSettings.SerializeToString,
            libreasr__pb2.Event.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def PreloadVoice(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/LibreASR.LibreASR/PreloadVoice",
            libreasr__pb2.AudioAndSettings.SerializeToString,
            libreasr__pb2.VoiceClip.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def Translate(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/LibreASR.LibreASR/Translate",
            libreasr__pb2.Text.SerializeToString,
            libreasr__pb2.Text.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
