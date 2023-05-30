# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: multiscope/protos/ui.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from multiscope.protos import tree_pb2 as multiscope_dot_protos_dot_tree__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1amultiscope/protos/ui.proto\x12\nmultiscope\x1a\x1cmultiscope/protos/tree.proto\"\x0b\n\tWorkerAck\"\'\n\x07\x43onnect\x12\x0e\n\x06scheme\x18\x01 \x01(\t\x12\x0c\n\x04host\x18\x02 \x01(\t\"\x17\n\x04Pull\x12\x0f\n\x07\x61\x63tives\x18\x01 \x03(\r\"a\n\x05Panel\x12\n\n\x02id\x18\x01 \x01(\r\x12#\n\x05paths\x18\x02 \x03(\x0b\x32\x14.multiscope.NodePath\x12\x15\n\rtransferables\x18\x03 \x03(\r\x12\x10\n\x08renderer\x18\x04 \x01(\t\"B\n\x0bStyleChange\x12\r\n\x05theme\x18\x01 \x01(\t\x12\x12\n\nfontFamily\x18\x02 \x01(\t\x12\x10\n\x08\x66ontSize\x18\x03 \x01(\x01\",\n\x0b\x45lementSize\x12\r\n\x05width\x18\n \x01(\x05\x12\x0e\n\x06height\x18\x0b \x01(\x05\"K\n\x0cParentResize\x12\x0f\n\x07panelID\x18\x01 \x01(\r\x12*\n\tchildSize\x18\n \x01(\x0b\x32\x17.multiscope.ElementSize\"h\n\x07UIEvent\x12(\n\x05style\x18\n \x01(\x0b\x32\x17.multiscope.StyleChangeH\x00\x12*\n\x06resize\x18\x0b \x01(\x0b\x32\x18.multiscope.ParentResizeH\x00\x42\x07\n\x05\x65vent\"a\n\rRegisterPanel\x12 \n\x05panel\x18\x01 \x01(\x0b\x32\x11.multiscope.Panel\x12.\n\rpreferredSize\x18\x02 \x01(\x0b\x32\x17.multiscope.ElementSize\"\xbd\x01\n\x08ToPuller\x12 \n\x04pull\x18\x01 \x01(\x0b\x32\x10.multiscope.PullH\x00\x12\x32\n\rregisterPanel\x18\x02 \x01(\x0b\x32\x19.multiscope.RegisterPanelH\x00\x12,\n\x0funregisterPanel\x18\x03 \x01(\x0b\x32\x11.multiscope.PanelH\x00\x12$\n\x05\x65vent\x18\x04 \x01(\x0b\x32\x13.multiscope.UIEventH\x00\x42\x07\n\x05query\"0\n\tPanelData\x12#\n\x05nodes\x18\x01 \x03(\x0b\x32\x14.multiscope.NodeData\"\x8f\x01\n\x0b\x44isplayData\x12\x0b\n\x03\x65rr\x18\x01 \x01(\t\x12/\n\x04\x64\x61ta\x18\x02 \x03(\x0b\x32!.multiscope.DisplayData.DataEntry\x1a\x42\n\tDataEntry\x12\x0b\n\x03key\x18\x01 \x01(\r\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.multiscope.PanelData:\x02\x38\x01\x42\x1fZ\x1dmultiscope/protos/ui_go_protob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'multiscope.protos.ui_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\035multiscope/protos/ui_go_proto'
  _DISPLAYDATA_DATAENTRY._options = None
  _DISPLAYDATA_DATAENTRY._serialized_options = b'8\001'
  _WORKERACK._serialized_start=72
  _WORKERACK._serialized_end=83
  _CONNECT._serialized_start=85
  _CONNECT._serialized_end=124
  _PULL._serialized_start=126
  _PULL._serialized_end=149
  _PANEL._serialized_start=151
  _PANEL._serialized_end=248
  _STYLECHANGE._serialized_start=250
  _STYLECHANGE._serialized_end=316
  _ELEMENTSIZE._serialized_start=318
  _ELEMENTSIZE._serialized_end=362
  _PARENTRESIZE._serialized_start=364
  _PARENTRESIZE._serialized_end=439
  _UIEVENT._serialized_start=441
  _UIEVENT._serialized_end=545
  _REGISTERPANEL._serialized_start=547
  _REGISTERPANEL._serialized_end=644
  _TOPULLER._serialized_start=647
  _TOPULLER._serialized_end=836
  _PANELDATA._serialized_start=838
  _PANELDATA._serialized_end=886
  _DISPLAYDATA._serialized_start=889
  _DISPLAYDATA._serialized_end=1032
  _DISPLAYDATA_DATAENTRY._serialized_start=966
  _DISPLAYDATA_DATAENTRY._serialized_end=1032
# @@protoc_insertion_point(module_scope)
