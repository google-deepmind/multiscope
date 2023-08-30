# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: multiscope/protos/text.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from multiscope.protos import tree_pb2 as multiscope_dot_protos_dot_tree__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1cmultiscope/protos/text.proto\x12\x0fmultiscope.text\x1a\x1cmultiscope/protos/tree.proto\"Q\n\x06Writer\x12#\n\x07tree_id\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\x12\"\n\x04path\x18\x02 \x01(\x0b\x32\x14.multiscope.NodePath\"U\n\nHTMLWriter\x12#\n\x07tree_id\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\x12\"\n\x04path\x18\x02 \x01(\x0b\x32\x14.multiscope.NodePath\"[\n\x10NewWriterRequest\x12#\n\x07tree_id\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\x12\"\n\x04path\x18\x02 \x01(\x0b\x32\x14.multiscope.NodePath\"<\n\x11NewWriterResponse\x12\'\n\x06writer\x18\x01 \x01(\x0b\x32\x17.multiscope.text.Writer\"_\n\x14NewHTMLWriterRequest\x12#\n\x07tree_id\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\x12\"\n\x04path\x18\x02 \x01(\x0b\x32\x14.multiscope.NodePath\"D\n\x15NewHTMLWriterResponse\x12+\n\x06writer\x18\x01 \x01(\x0b\x32\x1b.multiscope.text.HTMLWriter\"E\n\x0cWriteRequest\x12\'\n\x06writer\x18\x01 \x01(\x0b\x32\x17.multiscope.text.Writer\x12\x0c\n\x04text\x18\x02 \x01(\t\"\x0f\n\rWriteResponse\"M\n\x10WriteHTMLRequest\x12+\n\x06writer\x18\x01 \x01(\x0b\x32\x1b.multiscope.text.HTMLWriter\x12\x0c\n\x04html\x18\x02 \x01(\t\"\x13\n\x11WriteHTMLResponse\"K\n\x0fWriteCSSRequest\x12+\n\x06writer\x18\x01 \x01(\x0b\x32\x1b.multiscope.text.HTMLWriter\x12\x0b\n\x03\x63ss\x18\x02 \x01(\t\"\x12\n\x10WriteCSSResponse2\xb1\x03\n\x04Text\x12T\n\tNewWriter\x12!.multiscope.text.NewWriterRequest\x1a\".multiscope.text.NewWriterResponse\"\x00\x12`\n\rNewHTMLWriter\x12%.multiscope.text.NewHTMLWriterRequest\x1a&.multiscope.text.NewHTMLWriterResponse\"\x00\x12H\n\x05Write\x12\x1d.multiscope.text.WriteRequest\x1a\x1e.multiscope.text.WriteResponse\"\x00\x12T\n\tWriteHTML\x12!.multiscope.text.WriteHTMLRequest\x1a\".multiscope.text.WriteHTMLResponse\"\x00\x12Q\n\x08WriteCSS\x12 .multiscope.text.WriteCSSRequest\x1a!.multiscope.text.WriteCSSResponse\"\x00\x42!Z\x1fmultiscope/protos/text_go_protob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'multiscope.protos.text_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\037multiscope/protos/text_go_proto'
  _WRITER._serialized_start=79
  _WRITER._serialized_end=160
  _HTMLWRITER._serialized_start=162
  _HTMLWRITER._serialized_end=247
  _NEWWRITERREQUEST._serialized_start=249
  _NEWWRITERREQUEST._serialized_end=340
  _NEWWRITERRESPONSE._serialized_start=342
  _NEWWRITERRESPONSE._serialized_end=402
  _NEWHTMLWRITERREQUEST._serialized_start=404
  _NEWHTMLWRITERREQUEST._serialized_end=499
  _NEWHTMLWRITERRESPONSE._serialized_start=501
  _NEWHTMLWRITERRESPONSE._serialized_end=569
  _WRITEREQUEST._serialized_start=571
  _WRITEREQUEST._serialized_end=640
  _WRITERESPONSE._serialized_start=642
  _WRITERESPONSE._serialized_end=657
  _WRITEHTMLREQUEST._serialized_start=659
  _WRITEHTMLREQUEST._serialized_end=736
  _WRITEHTMLRESPONSE._serialized_start=738
  _WRITEHTMLRESPONSE._serialized_end=757
  _WRITECSSREQUEST._serialized_start=759
  _WRITECSSREQUEST._serialized_end=834
  _WRITECSSRESPONSE._serialized_start=836
  _WRITECSSRESPONSE._serialized_end=854
  _TEXT._serialized_start=857
  _TEXT._serialized_end=1290
# @@protoc_insertion_point(module_scope)
