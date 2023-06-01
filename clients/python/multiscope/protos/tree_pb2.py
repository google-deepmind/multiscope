# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: multiscope/protos/tree.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1cmultiscope/protos/tree.proto\x12\nmultiscope\x1a\x19google/protobuf/any.proto\"\x18\n\x06TreeID\x12\x0e\n\x06treeID\x18\x01 \x01(\x03\"\x18\n\x08NodePath\x12\x0c\n\x04path\x18\x01 \x03(\t\"\x8f\x01\n\x04Node\x12\r\n\x05\x65rror\x18\x01 \x01(\t\x12\"\n\x04path\x18\x02 \x01(\x0b\x32\x14.multiscope.NodePath\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\"\n\x08\x63hildren\x18\x04 \x03(\x0b\x32\x10.multiscope.Node\x12\x0c\n\x04mime\x18\x05 \x01(\t\x12\x14\n\x0chas_children\x18\x06 \x01(\x08\"\x98\x01\n\x08NodeData\x12\r\n\x05\x65rror\x18\x01 \x01(\t\x12\"\n\x04path\x18\x02 \x01(\x0b\x32\x14.multiscope.NodePath\x12\x0c\n\x04tick\x18\x03 \x01(\r\x12\x11\n\x03raw\x18\x06 \x01(\x0c\x42\x02\x08\x01H\x00\x12\"\n\x02pb\x18\x07 \x01(\x0b\x32\x14.google.protobuf.AnyH\x00\x12\x0c\n\x04mime\x18\x08 \x01(\tB\x06\n\x04\x64\x61ta\"R\n\x05\x45vent\x12\"\n\x04path\x18\x01 \x01(\x0b\x32\x14.multiscope.NodePath\x12%\n\x07payload\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any\"\\\n\x11NodeStructRequest\x12\"\n\x06treeID\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\x12#\n\x05paths\x18\x02 \x03(\x0b\x32\x14.multiscope.NodePath\"2\n\x0fNodeStructReply\x12\x1f\n\x05nodes\x18\x01 \x03(\x0b\x32\x10.multiscope.Node\"C\n\x0b\x44\x61taRequest\x12\"\n\x04path\x18\x01 \x01(\x0b\x32\x14.multiscope.NodePath\x12\x10\n\x08lastTick\x18\x02 \x01(\r\"\\\n\x0fNodeDataRequest\x12\"\n\x06treeID\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\x12%\n\x04reqs\x18\x02 \x03(\x0b\x32\x17.multiscope.DataRequest\"8\n\rNodeDataReply\x12\'\n\tnode_data\x18\x01 \x03(\x0b\x32\x14.multiscope.NodeData\"Z\n\x11SendEventsRequest\x12\"\n\x06treeID\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\x12!\n\x06\x65vents\x18\x02 \x03(\x0b\x32\x11.multiscope.Event\"9\n\x13StreamEventsRequest\x12\"\n\x06treeID\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\"!\n\x0fSendEventsReply\x12\x0e\n\x06\x65rrors\x18\x01 \x03(\t\"8\n\x12\x41\x63tivePathsRequest\x12\"\n\x06treeID\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\"7\n\x10\x41\x63tivePathsReply\x12#\n\x05paths\x18\x01 \x03(\x0b\x32\x14.multiscope.NodePath\"7\n\x11ResetStateRequest\x12\"\n\x06treeID\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\"\x11\n\x0fResetStateReply\"W\n\rDeleteRequest\x12\"\n\x06treeID\x18\x01 \x01(\x0b\x32\x12.multiscope.TreeID\x12\"\n\x04path\x18\x02 \x01(\x0b\x32\x14.multiscope.NodePath\"\r\n\x0b\x44\x65leteReply2\x8f\x04\n\x04Tree\x12M\n\rGetNodeStruct\x12\x1d.multiscope.NodeStructRequest\x1a\x1b.multiscope.NodeStructReply\"\x00\x12G\n\x0bGetNodeData\x12\x1b.multiscope.NodeDataRequest\x1a\x19.multiscope.NodeDataReply\"\x00\x12J\n\nSendEvents\x12\x1d.multiscope.SendEventsRequest\x1a\x1b.multiscope.SendEventsReply\"\x00\x12J\n\nResetState\x12\x1d.multiscope.ResetStateRequest\x1a\x1b.multiscope.ResetStateReply\"\x00\x12O\n\x0b\x41\x63tivePaths\x12\x1e.multiscope.ActivePathsRequest\x1a\x1c.multiscope.ActivePathsReply\"\x00\x30\x01\x12\x46\n\x0cStreamEvents\x12\x1f.multiscope.StreamEventsRequest\x1a\x11.multiscope.Event\"\x00\x30\x01\x12>\n\x06\x44\x65lete\x12\x19.multiscope.DeleteRequest\x1a\x17.multiscope.DeleteReply\"\x00\x42!Z\x1fmultiscope/protos/tree_go_protob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'multiscope.protos.tree_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\037multiscope/protos/tree_go_proto'
  _NODEDATA.fields_by_name['raw']._options = None
  _NODEDATA.fields_by_name['raw']._serialized_options = b'\010\001'
  _TREEID._serialized_start=71
  _TREEID._serialized_end=95
  _NODEPATH._serialized_start=97
  _NODEPATH._serialized_end=121
  _NODE._serialized_start=124
  _NODE._serialized_end=267
  _NODEDATA._serialized_start=270
  _NODEDATA._serialized_end=422
  _EVENT._serialized_start=424
  _EVENT._serialized_end=506
  _NODESTRUCTREQUEST._serialized_start=508
  _NODESTRUCTREQUEST._serialized_end=600
  _NODESTRUCTREPLY._serialized_start=602
  _NODESTRUCTREPLY._serialized_end=652
  _DATAREQUEST._serialized_start=654
  _DATAREQUEST._serialized_end=721
  _NODEDATAREQUEST._serialized_start=723
  _NODEDATAREQUEST._serialized_end=815
  _NODEDATAREPLY._serialized_start=817
  _NODEDATAREPLY._serialized_end=873
  _SENDEVENTSREQUEST._serialized_start=875
  _SENDEVENTSREQUEST._serialized_end=965
  _STREAMEVENTSREQUEST._serialized_start=967
  _STREAMEVENTSREQUEST._serialized_end=1024
  _SENDEVENTSREPLY._serialized_start=1026
  _SENDEVENTSREPLY._serialized_end=1059
  _ACTIVEPATHSREQUEST._serialized_start=1061
  _ACTIVEPATHSREQUEST._serialized_end=1117
  _ACTIVEPATHSREPLY._serialized_start=1119
  _ACTIVEPATHSREPLY._serialized_end=1174
  _RESETSTATEREQUEST._serialized_start=1176
  _RESETSTATEREQUEST._serialized_end=1231
  _RESETSTATEREPLY._serialized_start=1233
  _RESETSTATEREPLY._serialized_end=1250
  _DELETEREQUEST._serialized_start=1252
  _DELETEREQUEST._serialized_end=1339
  _DELETEREPLY._serialized_start=1341
  _DELETEREPLY._serialized_end=1354
  _TREE._serialized_start=1357
  _TREE._serialized_end=1884
# @@protoc_insertion_point(module_scope)
