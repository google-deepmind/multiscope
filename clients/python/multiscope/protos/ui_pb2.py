# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ui.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import tree_pb2 as tree__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x08ui.proto\x12\nmultiscope\x1a\ntree.proto\"\x0b\n\tWorkerAck\"\'\n\x07\x43onnect\x12\x0e\n\x06scheme\x18\x01 \x01(\t\x12\x0c\n\x04host\x18\x02 \x01(\t\"\x17\n\x04Pull\x12\x0f\n\x07\x61\x63tives\x18\x01 \x03(\r\"a\n\x05Panel\x12\n\n\x02id\x18\x01 \x01(\r\x12#\n\x05paths\x18\x02 \x03(\x0b\x32\x14.multiscope.NodePath\x12\x15\n\rtransferables\x18\x03 \x03(\r\x12\x10\n\x08renderer\x18\x04 \x01(\t\"B\n\x0bStyleChange\x12\r\n\x05theme\x18\x01 \x01(\t\x12\x12\n\nfontFamily\x18\x02 \x01(\t\x12\x10\n\x08\x66ontSize\x18\x03 \x01(\x01\"\x8b\x01\n\x08ToPuller\x12 \n\x04pull\x18\x01 \x01(\x0b\x32\x10.multiscope.PullH\x00\x12*\n\rregisterPanel\x18\x02 \x01(\x0b\x32\x11.multiscope.PanelH\x00\x12(\n\x05style\x18\x03 \x01(\x0b\x32\x17.multiscope.StyleChangeH\x00\x42\x07\n\x05query\"0\n\tPanelData\x12#\n\x05nodes\x18\x01 \x03(\x0b\x32\x14.multiscope.NodeData\"\x8f\x01\n\x0b\x44isplayData\x12\x0b\n\x03\x65rr\x18\x01 \x01(\t\x12/\n\x04\x64\x61ta\x18\x02 \x03(\x0b\x32!.multiscope.DisplayData.DataEntry\x1a\x42\n\tDataEntry\x12\x0b\n\x03key\x18\x01 \x01(\r\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.multiscope.PanelData:\x02\x38\x01\x42\x1fZ\x1dmultiscope/protos/ui_go_protob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ui_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\035multiscope/protos/ui_go_proto'
  _DISPLAYDATA_DATAENTRY._options = None
  _DISPLAYDATA_DATAENTRY._serialized_options = b'8\001'
  _WORKERACK._serialized_start=36
  _WORKERACK._serialized_end=47
  _CONNECT._serialized_start=49
  _CONNECT._serialized_end=88
  _PULL._serialized_start=90
  _PULL._serialized_end=113
  _PANEL._serialized_start=115
  _PANEL._serialized_end=212
  _STYLECHANGE._serialized_start=214
  _STYLECHANGE._serialized_end=280
  _TOPULLER._serialized_start=283
  _TOPULLER._serialized_end=422
  _PANELDATA._serialized_start=424
  _PANELDATA._serialized_end=472
  _DISPLAYDATA._serialized_start=475
  _DISPLAYDATA._serialized_end=618
  _DISPLAYDATA_DATAENTRY._serialized_start=552
  _DISPLAYDATA_DATAENTRY._serialized_end=618
# @@protoc_insertion_point(module_scope)