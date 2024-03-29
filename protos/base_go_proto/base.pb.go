// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Protocol buffers & services for creating and using basic writers and groups.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.30.0
// 	protoc        v3.21.12
// source: base.proto

package base_go_proto

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	anypb "google.golang.org/protobuf/types/known/anypb"
	tree_go_proto "multiscope/protos/tree_go_proto"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// Parent node in the Multiscope tree.
type Group struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	TreeId *tree_go_proto.TreeID `protobuf:"bytes,1,opt,name=tree_id,json=treeId,proto3" json:"tree_id,omitempty"`
	// Path in the tree.
	Path *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
}

func (x *Group) Reset() {
	*x = Group{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Group) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Group) ProtoMessage() {}

func (x *Group) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Group.ProtoReflect.Descriptor instead.
func (*Group) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{0}
}

func (x *Group) GetTreeId() *tree_go_proto.TreeID {
	if x != nil {
		return x.TreeId
	}
	return nil
}

func (x *Group) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

// Request to create a new ProtoWriter in the tree.
type NewGroupRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	TreeId *tree_go_proto.TreeID   `protobuf:"bytes,1,opt,name=tree_id,json=treeId,proto3" json:"tree_id,omitempty"`
	Path   *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
}

func (x *NewGroupRequest) Reset() {
	*x = NewGroupRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewGroupRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewGroupRequest) ProtoMessage() {}

func (x *NewGroupRequest) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NewGroupRequest.ProtoReflect.Descriptor instead.
func (*NewGroupRequest) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{1}
}

func (x *NewGroupRequest) GetTreeId() *tree_go_proto.TreeID {
	if x != nil {
		return x.TreeId
	}
	return nil
}

func (x *NewGroupRequest) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

// Response after creating a new group in the tree.
type NewGroupResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Grp *Group `protobuf:"bytes,1,opt,name=grp,proto3" json:"grp,omitempty"`
}

func (x *NewGroupResponse) Reset() {
	*x = NewGroupResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewGroupResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewGroupResponse) ProtoMessage() {}

func (x *NewGroupResponse) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NewGroupResponse.ProtoReflect.Descriptor instead.
func (*NewGroupResponse) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{2}
}

func (x *NewGroupResponse) GetGrp() *Group {
	if x != nil {
		return x.Grp
	}
	return nil
}

// Proto writer in the Multiscope tree.
type ProtoWriter struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	TreeId *tree_go_proto.TreeID `protobuf:"bytes,1,opt,name=tree_id,json=treeId,proto3" json:"tree_id,omitempty"`
	// Path in the tree.
	Path *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
}

func (x *ProtoWriter) Reset() {
	*x = ProtoWriter{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProtoWriter) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProtoWriter) ProtoMessage() {}

func (x *ProtoWriter) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProtoWriter.ProtoReflect.Descriptor instead.
func (*ProtoWriter) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{3}
}

func (x *ProtoWriter) GetTreeId() *tree_go_proto.TreeID {
	if x != nil {
		return x.TreeId
	}
	return nil
}

func (x *ProtoWriter) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

// Request to create a new ProtoWriter in the tree.
type NewProtoWriterRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	TreeId *tree_go_proto.TreeID `protobuf:"bytes,1,opt,name=tree_id,json=treeId,proto3" json:"tree_id,omitempty"`
	// Path in the tree.
	Path *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
	// An instance of the protocol buffers this writer will write.
	Proto *anypb.Any `protobuf:"bytes,3,opt,name=proto,proto3" json:"proto,omitempty"`
}

func (x *NewProtoWriterRequest) Reset() {
	*x = NewProtoWriterRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewProtoWriterRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewProtoWriterRequest) ProtoMessage() {}

func (x *NewProtoWriterRequest) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NewProtoWriterRequest.ProtoReflect.Descriptor instead.
func (*NewProtoWriterRequest) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{4}
}

func (x *NewProtoWriterRequest) GetTreeId() *tree_go_proto.TreeID {
	if x != nil {
		return x.TreeId
	}
	return nil
}

func (x *NewProtoWriterRequest) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

func (x *NewProtoWriterRequest) GetProto() *anypb.Any {
	if x != nil {
		return x.Proto
	}
	return nil
}

// Response after creating a new ProtoWriter in the tree.
type NewProtoWriterResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Writer *ProtoWriter `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
}

func (x *NewProtoWriterResponse) Reset() {
	*x = NewProtoWriterResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[5]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewProtoWriterResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewProtoWriterResponse) ProtoMessage() {}

func (x *NewProtoWriterResponse) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[5]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NewProtoWriterResponse.ProtoReflect.Descriptor instead.
func (*NewProtoWriterResponse) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{5}
}

func (x *NewProtoWriterResponse) GetWriter() *ProtoWriter {
	if x != nil {
		return x.Writer
	}
	return nil
}

// Request to write a proto.
type WriteProtoRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Writer to write the data to.
	Writer *ProtoWriter `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
	// Data to write.
	Proto *anypb.Any `protobuf:"bytes,2,opt,name=proto,proto3" json:"proto,omitempty"`
}

func (x *WriteProtoRequest) Reset() {
	*x = WriteProtoRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[6]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteProtoRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteProtoRequest) ProtoMessage() {}

func (x *WriteProtoRequest) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[6]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WriteProtoRequest.ProtoReflect.Descriptor instead.
func (*WriteProtoRequest) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{6}
}

func (x *WriteProtoRequest) GetWriter() *ProtoWriter {
	if x != nil {
		return x.Writer
	}
	return nil
}

func (x *WriteProtoRequest) GetProto() *anypb.Any {
	if x != nil {
		return x.Proto
	}
	return nil
}

type WriteProtoResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *WriteProtoResponse) Reset() {
	*x = WriteProtoResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[7]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteProtoResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteProtoResponse) ProtoMessage() {}

func (x *WriteProtoResponse) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[7]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WriteProtoResponse.ProtoReflect.Descriptor instead.
func (*WriteProtoResponse) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{7}
}

// Raw writer in the Multiscope tree.
type RawWriter struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	TreeId *tree_go_proto.TreeID `protobuf:"bytes,1,opt,name=tree_id,json=treeId,proto3" json:"tree_id,omitempty"`
	// Path in the tree.
	Path *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
}

func (x *RawWriter) Reset() {
	*x = RawWriter{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[8]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *RawWriter) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RawWriter) ProtoMessage() {}

func (x *RawWriter) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[8]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RawWriter.ProtoReflect.Descriptor instead.
func (*RawWriter) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{8}
}

func (x *RawWriter) GetTreeId() *tree_go_proto.TreeID {
	if x != nil {
		return x.TreeId
	}
	return nil
}

func (x *RawWriter) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

// Request to create a new RawWriter in the tree.
type NewRawWriterRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	TreeId *tree_go_proto.TreeID `protobuf:"bytes,1,opt,name=tree_id,json=treeId,proto3" json:"tree_id,omitempty"`
	// Path in the tree.
	Path *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
	// MIME type of the raw bytes to be written to this writer.
	Mime string `protobuf:"bytes,3,opt,name=mime,proto3" json:"mime,omitempty"`
}

func (x *NewRawWriterRequest) Reset() {
	*x = NewRawWriterRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[9]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewRawWriterRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewRawWriterRequest) ProtoMessage() {}

func (x *NewRawWriterRequest) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[9]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NewRawWriterRequest.ProtoReflect.Descriptor instead.
func (*NewRawWriterRequest) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{9}
}

func (x *NewRawWriterRequest) GetTreeId() *tree_go_proto.TreeID {
	if x != nil {
		return x.TreeId
	}
	return nil
}

func (x *NewRawWriterRequest) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

func (x *NewRawWriterRequest) GetMime() string {
	if x != nil {
		return x.Mime
	}
	return ""
}

// Response after creating a new RawWriter in the tree.
type NewRawWriterResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Writer *RawWriter `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
}

func (x *NewRawWriterResponse) Reset() {
	*x = NewRawWriterResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[10]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewRawWriterResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewRawWriterResponse) ProtoMessage() {}

func (x *NewRawWriterResponse) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[10]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NewRawWriterResponse.ProtoReflect.Descriptor instead.
func (*NewRawWriterResponse) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{10}
}

func (x *NewRawWriterResponse) GetWriter() *RawWriter {
	if x != nil {
		return x.Writer
	}
	return nil
}

// Request to write raw data.
type WriteRawRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Writer to write the data to.
	Writer *RawWriter `protobuf:"bytes,2,opt,name=writer,proto3" json:"writer,omitempty"`
	// Data to write.
	Data []byte `protobuf:"bytes,3,opt,name=data,proto3" json:"data,omitempty"`
}

func (x *WriteRawRequest) Reset() {
	*x = WriteRawRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[11]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteRawRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteRawRequest) ProtoMessage() {}

func (x *WriteRawRequest) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[11]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WriteRawRequest.ProtoReflect.Descriptor instead.
func (*WriteRawRequest) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{11}
}

func (x *WriteRawRequest) GetWriter() *RawWriter {
	if x != nil {
		return x.Writer
	}
	return nil
}

func (x *WriteRawRequest) GetData() []byte {
	if x != nil {
		return x.Data
	}
	return nil
}

type WriteRawResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *WriteRawResponse) Reset() {
	*x = WriteRawResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_base_proto_msgTypes[12]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteRawResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteRawResponse) ProtoMessage() {}

func (x *WriteRawResponse) ProtoReflect() protoreflect.Message {
	mi := &file_base_proto_msgTypes[12]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WriteRawResponse.ProtoReflect.Descriptor instead.
func (*WriteRawResponse) Descriptor() ([]byte, []int) {
	return file_base_proto_rawDescGZIP(), []int{12}
}

var File_base_proto protoreflect.FileDescriptor

var file_base_proto_rawDesc = []byte{
	0x0a, 0x0a, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0f, 0x6d, 0x75,
	0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x1a, 0x19, 0x67,
	0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f, 0x61,
	0x6e, 0x79, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x0a, 0x74, 0x72, 0x65, 0x65, 0x2e, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x22, 0x5e, 0x0a, 0x05, 0x47, 0x72, 0x6f, 0x75, 0x70, 0x12, 0x2b, 0x0a,
	0x07, 0x74, 0x72, 0x65, 0x65, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x12,
	0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x54, 0x72, 0x65, 0x65,
	0x49, 0x44, 0x52, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49, 0x64, 0x12, 0x28, 0x0a, 0x04, 0x70, 0x61,
	0x74, 0x68, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69,
	0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x50, 0x61, 0x74, 0x68, 0x52, 0x04,
	0x70, 0x61, 0x74, 0x68, 0x22, 0x68, 0x0a, 0x0f, 0x4e, 0x65, 0x77, 0x47, 0x72, 0x6f, 0x75, 0x70,
	0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x2b, 0x0a, 0x07, 0x74, 0x72, 0x65, 0x65, 0x5f,
	0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x12, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69,
	0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x54, 0x72, 0x65, 0x65, 0x49, 0x44, 0x52, 0x06, 0x74, 0x72,
	0x65, 0x65, 0x49, 0x64, 0x12, 0x28, 0x0a, 0x04, 0x70, 0x61, 0x74, 0x68, 0x18, 0x02, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e,
	0x4e, 0x6f, 0x64, 0x65, 0x50, 0x61, 0x74, 0x68, 0x52, 0x04, 0x70, 0x61, 0x74, 0x68, 0x22, 0x3c,
	0x0a, 0x10, 0x4e, 0x65, 0x77, 0x47, 0x72, 0x6f, 0x75, 0x70, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e,
	0x73, 0x65, 0x12, 0x28, 0x0a, 0x03, 0x67, 0x72, 0x70, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32,
	0x16, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73,
	0x65, 0x2e, 0x47, 0x72, 0x6f, 0x75, 0x70, 0x52, 0x03, 0x67, 0x72, 0x70, 0x22, 0x64, 0x0a, 0x0b,
	0x50, 0x72, 0x6f, 0x74, 0x6f, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x2b, 0x0a, 0x07, 0x74,
	0x72, 0x65, 0x65, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x12, 0x2e, 0x6d,
	0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x54, 0x72, 0x65, 0x65, 0x49, 0x44,
	0x52, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49, 0x64, 0x12, 0x28, 0x0a, 0x04, 0x70, 0x61, 0x74, 0x68,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63,
	0x6f, 0x70, 0x65, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x50, 0x61, 0x74, 0x68, 0x52, 0x04, 0x70, 0x61,
	0x74, 0x68, 0x22, 0x9a, 0x01, 0x0a, 0x15, 0x4e, 0x65, 0x77, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x2b, 0x0a, 0x07,
	0x74, 0x72, 0x65, 0x65, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x12, 0x2e,
	0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x54, 0x72, 0x65, 0x65, 0x49,
	0x44, 0x52, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49, 0x64, 0x12, 0x28, 0x0a, 0x04, 0x70, 0x61, 0x74,
	0x68, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73,
	0x63, 0x6f, 0x70, 0x65, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x50, 0x61, 0x74, 0x68, 0x52, 0x04, 0x70,
	0x61, 0x74, 0x68, 0x12, 0x2a, 0x0a, 0x05, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x18, 0x03, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x14, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x62, 0x75, 0x66, 0x2e, 0x41, 0x6e, 0x79, 0x52, 0x05, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22,
	0x4e, 0x0a, 0x16, 0x4e, 0x65, 0x77, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x57, 0x72, 0x69, 0x74, 0x65,
	0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x34, 0x0a, 0x06, 0x77, 0x72, 0x69,
	0x74, 0x65, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1c, 0x2e, 0x6d, 0x75, 0x6c, 0x74,
	0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x50, 0x72, 0x6f, 0x74,
	0x6f, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x22,
	0x75, 0x0a, 0x11, 0x57, 0x72, 0x69, 0x74, 0x65, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x52, 0x65, 0x71,
	0x75, 0x65, 0x73, 0x74, 0x12, 0x34, 0x0a, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x0b, 0x32, 0x1c, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70,
	0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x57, 0x72, 0x69, 0x74,
	0x65, 0x72, 0x52, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x2a, 0x0a, 0x05, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x67, 0x6f, 0x6f, 0x67,
	0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x41, 0x6e, 0x79, 0x52,
	0x05, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0x14, 0x0a, 0x12, 0x57, 0x72, 0x69, 0x74, 0x65, 0x50,
	0x72, 0x6f, 0x74, 0x6f, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x62, 0x0a, 0x09,
	0x52, 0x61, 0x77, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x2b, 0x0a, 0x07, 0x74, 0x72, 0x65,
	0x65, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x12, 0x2e, 0x6d, 0x75, 0x6c,
	0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x54, 0x72, 0x65, 0x65, 0x49, 0x44, 0x52, 0x06,
	0x74, 0x72, 0x65, 0x65, 0x49, 0x64, 0x12, 0x28, 0x0a, 0x04, 0x70, 0x61, 0x74, 0x68, 0x18, 0x02,
	0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70,
	0x65, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x50, 0x61, 0x74, 0x68, 0x52, 0x04, 0x70, 0x61, 0x74, 0x68,
	0x22, 0x80, 0x01, 0x0a, 0x13, 0x4e, 0x65, 0x77, 0x52, 0x61, 0x77, 0x57, 0x72, 0x69, 0x74, 0x65,
	0x72, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x2b, 0x0a, 0x07, 0x74, 0x72, 0x65, 0x65,
	0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x12, 0x2e, 0x6d, 0x75, 0x6c, 0x74,
	0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x54, 0x72, 0x65, 0x65, 0x49, 0x44, 0x52, 0x06, 0x74,
	0x72, 0x65, 0x65, 0x49, 0x64, 0x12, 0x28, 0x0a, 0x04, 0x70, 0x61, 0x74, 0x68, 0x18, 0x02, 0x20,
	0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65,
	0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x50, 0x61, 0x74, 0x68, 0x52, 0x04, 0x70, 0x61, 0x74, 0x68, 0x12,
	0x12, 0x0a, 0x04, 0x6d, 0x69, 0x6d, 0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6d,
	0x69, 0x6d, 0x65, 0x22, 0x4a, 0x0a, 0x14, 0x4e, 0x65, 0x77, 0x52, 0x61, 0x77, 0x57, 0x72, 0x69,
	0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x32, 0x0a, 0x06, 0x77,
	0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1a, 0x2e, 0x6d, 0x75,
	0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x52, 0x61,
	0x77, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x22,
	0x59, 0x0a, 0x0f, 0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x61, 0x77, 0x52, 0x65, 0x71, 0x75, 0x65,
	0x73, 0x74, 0x12, 0x32, 0x0a, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x02, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x1a, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e,
	0x62, 0x61, 0x73, 0x65, 0x2e, 0x52, 0x61, 0x77, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x06,
	0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x12, 0x0a, 0x04, 0x64, 0x61, 0x74, 0x61, 0x18, 0x03,
	0x20, 0x01, 0x28, 0x0c, 0x52, 0x04, 0x64, 0x61, 0x74, 0x61, 0x22, 0x12, 0x0a, 0x10, 0x57, 0x72,
	0x69, 0x74, 0x65, 0x52, 0x61, 0x77, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x32, 0xd0,
	0x03, 0x0a, 0x0b, 0x42, 0x61, 0x73, 0x65, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x73, 0x12, 0x51,
	0x0a, 0x08, 0x4e, 0x65, 0x77, 0x47, 0x72, 0x6f, 0x75, 0x70, 0x12, 0x20, 0x2e, 0x6d, 0x75, 0x6c,
	0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x4e, 0x65, 0x77,
	0x47, 0x72, 0x6f, 0x75, 0x70, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x21, 0x2e, 0x6d,
	0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x4e,
	0x65, 0x77, 0x47, 0x72, 0x6f, 0x75, 0x70, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22,
	0x00, 0x12, 0x63, 0x0a, 0x0e, 0x4e, 0x65, 0x77, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x57, 0x72, 0x69,
	0x74, 0x65, 0x72, 0x12, 0x26, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65,
	0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x4e, 0x65, 0x77, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x57, 0x72,
	0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x27, 0x2e, 0x6d, 0x75,
	0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x4e, 0x65,
	0x77, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70,
	0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12, 0x57, 0x0a, 0x0a, 0x57, 0x72, 0x69, 0x74, 0x65, 0x50,
	0x72, 0x6f, 0x74, 0x6f, 0x12, 0x22, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70,
	0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x50, 0x72, 0x6f, 0x74,
	0x6f, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x23, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69,
	0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65,
	0x50, 0x72, 0x6f, 0x74, 0x6f, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12,
	0x5d, 0x0a, 0x0c, 0x4e, 0x65, 0x77, 0x52, 0x61, 0x77, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12,
	0x24, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73,
	0x65, 0x2e, 0x4e, 0x65, 0x77, 0x52, 0x61, 0x77, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x25, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f,
	0x70, 0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x4e, 0x65, 0x77, 0x52, 0x61, 0x77, 0x57, 0x72,
	0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12, 0x51,
	0x0a, 0x08, 0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x61, 0x77, 0x12, 0x20, 0x2e, 0x6d, 0x75, 0x6c,
	0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x57, 0x72, 0x69,
	0x74, 0x65, 0x52, 0x61, 0x77, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x21, 0x2e, 0x6d,
	0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x62, 0x61, 0x73, 0x65, 0x2e, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x52, 0x61, 0x77, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22,
	0x00, 0x42, 0x21, 0x5a, 0x1f, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2f,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x62, 0x61, 0x73, 0x65, 0x5f, 0x67, 0x6f, 0x5f, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_base_proto_rawDescOnce sync.Once
	file_base_proto_rawDescData = file_base_proto_rawDesc
)

func file_base_proto_rawDescGZIP() []byte {
	file_base_proto_rawDescOnce.Do(func() {
		file_base_proto_rawDescData = protoimpl.X.CompressGZIP(file_base_proto_rawDescData)
	})
	return file_base_proto_rawDescData
}

var file_base_proto_msgTypes = make([]protoimpl.MessageInfo, 13)
var file_base_proto_goTypes = []interface{}{
	(*Group)(nil),                  // 0: multiscope.base.Group
	(*NewGroupRequest)(nil),        // 1: multiscope.base.NewGroupRequest
	(*NewGroupResponse)(nil),       // 2: multiscope.base.NewGroupResponse
	(*ProtoWriter)(nil),            // 3: multiscope.base.ProtoWriter
	(*NewProtoWriterRequest)(nil),  // 4: multiscope.base.NewProtoWriterRequest
	(*NewProtoWriterResponse)(nil), // 5: multiscope.base.NewProtoWriterResponse
	(*WriteProtoRequest)(nil),      // 6: multiscope.base.WriteProtoRequest
	(*WriteProtoResponse)(nil),     // 7: multiscope.base.WriteProtoResponse
	(*RawWriter)(nil),              // 8: multiscope.base.RawWriter
	(*NewRawWriterRequest)(nil),    // 9: multiscope.base.NewRawWriterRequest
	(*NewRawWriterResponse)(nil),   // 10: multiscope.base.NewRawWriterResponse
	(*WriteRawRequest)(nil),        // 11: multiscope.base.WriteRawRequest
	(*WriteRawResponse)(nil),       // 12: multiscope.base.WriteRawResponse
	(*tree_go_proto.TreeID)(nil),   // 13: multiscope.TreeID
	(*tree_go_proto.NodePath)(nil), // 14: multiscope.NodePath
	(*anypb.Any)(nil),              // 15: google.protobuf.Any
}
var file_base_proto_depIdxs = []int32{
	13, // 0: multiscope.base.Group.tree_id:type_name -> multiscope.TreeID
	14, // 1: multiscope.base.Group.path:type_name -> multiscope.NodePath
	13, // 2: multiscope.base.NewGroupRequest.tree_id:type_name -> multiscope.TreeID
	14, // 3: multiscope.base.NewGroupRequest.path:type_name -> multiscope.NodePath
	0,  // 4: multiscope.base.NewGroupResponse.grp:type_name -> multiscope.base.Group
	13, // 5: multiscope.base.ProtoWriter.tree_id:type_name -> multiscope.TreeID
	14, // 6: multiscope.base.ProtoWriter.path:type_name -> multiscope.NodePath
	13, // 7: multiscope.base.NewProtoWriterRequest.tree_id:type_name -> multiscope.TreeID
	14, // 8: multiscope.base.NewProtoWriterRequest.path:type_name -> multiscope.NodePath
	15, // 9: multiscope.base.NewProtoWriterRequest.proto:type_name -> google.protobuf.Any
	3,  // 10: multiscope.base.NewProtoWriterResponse.writer:type_name -> multiscope.base.ProtoWriter
	3,  // 11: multiscope.base.WriteProtoRequest.writer:type_name -> multiscope.base.ProtoWriter
	15, // 12: multiscope.base.WriteProtoRequest.proto:type_name -> google.protobuf.Any
	13, // 13: multiscope.base.RawWriter.tree_id:type_name -> multiscope.TreeID
	14, // 14: multiscope.base.RawWriter.path:type_name -> multiscope.NodePath
	13, // 15: multiscope.base.NewRawWriterRequest.tree_id:type_name -> multiscope.TreeID
	14, // 16: multiscope.base.NewRawWriterRequest.path:type_name -> multiscope.NodePath
	8,  // 17: multiscope.base.NewRawWriterResponse.writer:type_name -> multiscope.base.RawWriter
	8,  // 18: multiscope.base.WriteRawRequest.writer:type_name -> multiscope.base.RawWriter
	1,  // 19: multiscope.base.BaseWriters.NewGroup:input_type -> multiscope.base.NewGroupRequest
	4,  // 20: multiscope.base.BaseWriters.NewProtoWriter:input_type -> multiscope.base.NewProtoWriterRequest
	6,  // 21: multiscope.base.BaseWriters.WriteProto:input_type -> multiscope.base.WriteProtoRequest
	9,  // 22: multiscope.base.BaseWriters.NewRawWriter:input_type -> multiscope.base.NewRawWriterRequest
	11, // 23: multiscope.base.BaseWriters.WriteRaw:input_type -> multiscope.base.WriteRawRequest
	2,  // 24: multiscope.base.BaseWriters.NewGroup:output_type -> multiscope.base.NewGroupResponse
	5,  // 25: multiscope.base.BaseWriters.NewProtoWriter:output_type -> multiscope.base.NewProtoWriterResponse
	7,  // 26: multiscope.base.BaseWriters.WriteProto:output_type -> multiscope.base.WriteProtoResponse
	10, // 27: multiscope.base.BaseWriters.NewRawWriter:output_type -> multiscope.base.NewRawWriterResponse
	12, // 28: multiscope.base.BaseWriters.WriteRaw:output_type -> multiscope.base.WriteRawResponse
	24, // [24:29] is the sub-list for method output_type
	19, // [19:24] is the sub-list for method input_type
	19, // [19:19] is the sub-list for extension type_name
	19, // [19:19] is the sub-list for extension extendee
	0,  // [0:19] is the sub-list for field type_name
}

func init() { file_base_proto_init() }
func file_base_proto_init() {
	if File_base_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_base_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Group); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NewGroupRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NewGroupResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProtoWriter); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NewProtoWriterRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[5].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NewProtoWriterResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[6].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WriteProtoRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[7].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WriteProtoResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[8].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*RawWriter); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[9].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NewRawWriterRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[10].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NewRawWriterResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[11].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WriteRawRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_base_proto_msgTypes[12].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WriteRawResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_base_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   13,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_base_proto_goTypes,
		DependencyIndexes: file_base_proto_depIdxs,
		MessageInfos:      file_base_proto_msgTypes,
	}.Build()
	File_base_proto = out.File
	file_base_proto_rawDesc = nil
	file_base_proto_goTypes = nil
	file_base_proto_depIdxs = nil
}
