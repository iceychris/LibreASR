# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: libreasr.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name="libreasr.proto",
    package="LibreASR",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x0elibreasr.proto\x12\x08LibreASR"/\n\x05\x41udio\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\n\n\x02sr\x18\x02 \x01(\x05\x12\x0c\n\x04lang\x18\x03 \x01(\t"\x1a\n\nTranscript\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t"%\n\x0fTranscriptEvent\x12\x12\n\ntranscript\x18\x01 \x01(\t"#\n\rSentenceEvent\x12\x12\n\ntranscript\x18\x01 \x01(\t"b\n\x05\x45vent\x12\'\n\x02te\x18\x01 \x01(\x0b\x32\x19.LibreASR.TranscriptEventH\x00\x12%\n\x02se\x18\x02 \x01(\x0b\x32\x17.LibreASR.SentenceEventH\x00\x42\t\n\x07\x63ontent2}\n\x08LibreASR\x12\x35\n\nTranscribe\x12\x0f.LibreASR.Audio\x1a\x14.LibreASR.Transcript"\x00\x12:\n\x10TranscribeStream\x12\x0f.LibreASR.Audio\x1a\x0f.LibreASR.Event"\x00(\x01\x30\x01\x62\x06proto3',
)


_AUDIO = _descriptor.Descriptor(
    name="Audio",
    full_name="LibreASR.Audio",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="data",
            full_name="LibreASR.Audio.data",
            index=0,
            number=1,
            type=12,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"",
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="sr",
            full_name="LibreASR.Audio.sr",
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="lang",
            full_name="LibreASR.Audio.lang",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=28,
    serialized_end=75,
)


_TRANSCRIPT = _descriptor.Descriptor(
    name="Transcript",
    full_name="LibreASR.Transcript",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="data",
            full_name="LibreASR.Transcript.data",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=77,
    serialized_end=103,
)


_TRANSCRIPTEVENT = _descriptor.Descriptor(
    name="TranscriptEvent",
    full_name="LibreASR.TranscriptEvent",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="transcript",
            full_name="LibreASR.TranscriptEvent.transcript",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=105,
    serialized_end=142,
)


_SENTENCEEVENT = _descriptor.Descriptor(
    name="SentenceEvent",
    full_name="LibreASR.SentenceEvent",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="transcript",
            full_name="LibreASR.SentenceEvent.transcript",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=144,
    serialized_end=179,
)


_EVENT = _descriptor.Descriptor(
    name="Event",
    full_name="LibreASR.Event",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="te",
            full_name="LibreASR.Event.te",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="se",
            full_name="LibreASR.Event.se",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[
        _descriptor.OneofDescriptor(
            name="content",
            full_name="LibreASR.Event.content",
            index=0,
            containing_type=None,
            create_key=_descriptor._internal_create_key,
            fields=[],
        ),
    ],
    serialized_start=181,
    serialized_end=279,
)

_EVENT.fields_by_name["te"].message_type = _TRANSCRIPTEVENT
_EVENT.fields_by_name["se"].message_type = _SENTENCEEVENT
_EVENT.oneofs_by_name["content"].fields.append(_EVENT.fields_by_name["te"])
_EVENT.fields_by_name["te"].containing_oneof = _EVENT.oneofs_by_name["content"]
_EVENT.oneofs_by_name["content"].fields.append(_EVENT.fields_by_name["se"])
_EVENT.fields_by_name["se"].containing_oneof = _EVENT.oneofs_by_name["content"]
DESCRIPTOR.message_types_by_name["Audio"] = _AUDIO
DESCRIPTOR.message_types_by_name["Transcript"] = _TRANSCRIPT
DESCRIPTOR.message_types_by_name["TranscriptEvent"] = _TRANSCRIPTEVENT
DESCRIPTOR.message_types_by_name["SentenceEvent"] = _SENTENCEEVENT
DESCRIPTOR.message_types_by_name["Event"] = _EVENT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Audio = _reflection.GeneratedProtocolMessageType(
    "Audio",
    (_message.Message,),
    {
        "DESCRIPTOR": _AUDIO,
        "__module__": "libreasr_pb2"
        # @@protoc_insertion_point(class_scope:LibreASR.Audio)
    },
)
_sym_db.RegisterMessage(Audio)

Transcript = _reflection.GeneratedProtocolMessageType(
    "Transcript",
    (_message.Message,),
    {
        "DESCRIPTOR": _TRANSCRIPT,
        "__module__": "libreasr_pb2"
        # @@protoc_insertion_point(class_scope:LibreASR.Transcript)
    },
)
_sym_db.RegisterMessage(Transcript)

TranscriptEvent = _reflection.GeneratedProtocolMessageType(
    "TranscriptEvent",
    (_message.Message,),
    {
        "DESCRIPTOR": _TRANSCRIPTEVENT,
        "__module__": "libreasr_pb2"
        # @@protoc_insertion_point(class_scope:LibreASR.TranscriptEvent)
    },
)
_sym_db.RegisterMessage(TranscriptEvent)

SentenceEvent = _reflection.GeneratedProtocolMessageType(
    "SentenceEvent",
    (_message.Message,),
    {
        "DESCRIPTOR": _SENTENCEEVENT,
        "__module__": "libreasr_pb2"
        # @@protoc_insertion_point(class_scope:LibreASR.SentenceEvent)
    },
)
_sym_db.RegisterMessage(SentenceEvent)

Event = _reflection.GeneratedProtocolMessageType(
    "Event",
    (_message.Message,),
    {
        "DESCRIPTOR": _EVENT,
        "__module__": "libreasr_pb2"
        # @@protoc_insertion_point(class_scope:LibreASR.Event)
    },
)
_sym_db.RegisterMessage(Event)


_LIBREASR = _descriptor.ServiceDescriptor(
    name="LibreASR",
    full_name="LibreASR.LibreASR",
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=281,
    serialized_end=406,
    methods=[
        _descriptor.MethodDescriptor(
            name="Transcribe",
            full_name="LibreASR.LibreASR.Transcribe",
            index=0,
            containing_service=None,
            input_type=_AUDIO,
            output_type=_TRANSCRIPT,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name="TranscribeStream",
            full_name="LibreASR.LibreASR.TranscribeStream",
            index=1,
            containing_service=None,
            input_type=_AUDIO,
            output_type=_EVENT,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
)
_sym_db.RegisterServiceDescriptor(_LIBREASR)

DESCRIPTOR.services_by_name["LibreASR"] = _LIBREASR

# @@protoc_insertion_point(module_scope)
