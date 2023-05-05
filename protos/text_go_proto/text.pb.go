// Protocol buffer to stream text data.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.30.0
// 	protoc        v3.21.12
// source: text.proto

package text_go_proto

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
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

type Writer struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	TreeID *tree_go_proto.TreeID `protobuf:"bytes,1,opt,name=treeID,proto3" json:"treeID,omitempty"`
	// Path in the Multiscope tree.
	Path *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
}

func (x *Writer) Reset() {
	*x = Writer{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Writer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Writer) ProtoMessage() {}

func (x *Writer) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Writer.ProtoReflect.Descriptor instead.
func (*Writer) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{0}
}

func (x *Writer) GetTreeID() *tree_go_proto.TreeID {
	if x != nil {
		return x.TreeID
	}
	return nil
}

func (x *Writer) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

type HTMLWriter struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	TreeID *tree_go_proto.TreeID `protobuf:"bytes,1,opt,name=treeID,proto3" json:"treeID,omitempty"`
	// Path in the Multiscope tree.
	Path *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
}

func (x *HTMLWriter) Reset() {
	*x = HTMLWriter{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *HTMLWriter) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HTMLWriter) ProtoMessage() {}

func (x *HTMLWriter) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use HTMLWriter.ProtoReflect.Descriptor instead.
func (*HTMLWriter) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{1}
}

func (x *HTMLWriter) GetTreeID() *tree_go_proto.TreeID {
	if x != nil {
		return x.TreeID
	}
	return nil
}

func (x *HTMLWriter) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

// Request to create a new (raw) text writer in the tree.
type NewWriterRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	TreeID *tree_go_proto.TreeID   `protobuf:"bytes,1,opt,name=treeID,proto3" json:"treeID,omitempty"`
	Path   *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
}

func (x *NewWriterRequest) Reset() {
	*x = NewWriterRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewWriterRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewWriterRequest) ProtoMessage() {}

func (x *NewWriterRequest) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NewWriterRequest.ProtoReflect.Descriptor instead.
func (*NewWriterRequest) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{2}
}

func (x *NewWriterRequest) GetTreeID() *tree_go_proto.TreeID {
	if x != nil {
		return x.TreeID
	}
	return nil
}

func (x *NewWriterRequest) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

// Response after creating a new (raw) text writer in the tree.
type NewWriterResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Writer *Writer `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
}

func (x *NewWriterResponse) Reset() {
	*x = NewWriterResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewWriterResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewWriterResponse) ProtoMessage() {}

func (x *NewWriterResponse) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NewWriterResponse.ProtoReflect.Descriptor instead.
func (*NewWriterResponse) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{3}
}

func (x *NewWriterResponse) GetWriter() *Writer {
	if x != nil {
		return x.Writer
	}
	return nil
}

// Request to create a new HTML writer in the tree.
type NewHTMLWriterRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	TreeID *tree_go_proto.TreeID   `protobuf:"bytes,1,opt,name=treeID,proto3" json:"treeID,omitempty"`
	Path   *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
}

func (x *NewHTMLWriterRequest) Reset() {
	*x = NewHTMLWriterRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewHTMLWriterRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewHTMLWriterRequest) ProtoMessage() {}

func (x *NewHTMLWriterRequest) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NewHTMLWriterRequest.ProtoReflect.Descriptor instead.
func (*NewHTMLWriterRequest) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{4}
}

func (x *NewHTMLWriterRequest) GetTreeID() *tree_go_proto.TreeID {
	if x != nil {
		return x.TreeID
	}
	return nil
}

func (x *NewHTMLWriterRequest) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

// Response after creating a new HTML writer in the tree.
type NewHTMLWriterResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Writer *HTMLWriter `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
}

func (x *NewHTMLWriterResponse) Reset() {
	*x = NewHTMLWriterResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[5]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewHTMLWriterResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewHTMLWriterResponse) ProtoMessage() {}

func (x *NewHTMLWriterResponse) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[5]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NewHTMLWriterResponse.ProtoReflect.Descriptor instead.
func (*NewHTMLWriterResponse) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{5}
}

func (x *NewHTMLWriterResponse) GetWriter() *HTMLWriter {
	if x != nil {
		return x.Writer
	}
	return nil
}

// Request to write text to a Writer.
type WriteRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Writer *Writer `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
	Text   string  `protobuf:"bytes,2,opt,name=text,proto3" json:"text,omitempty"`
}

func (x *WriteRequest) Reset() {
	*x = WriteRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[6]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteRequest) ProtoMessage() {}

func (x *WriteRequest) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[6]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WriteRequest.ProtoReflect.Descriptor instead.
func (*WriteRequest) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{6}
}

func (x *WriteRequest) GetWriter() *Writer {
	if x != nil {
		return x.Writer
	}
	return nil
}

func (x *WriteRequest) GetText() string {
	if x != nil {
		return x.Text
	}
	return ""
}

type WriteResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *WriteResponse) Reset() {
	*x = WriteResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[7]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteResponse) ProtoMessage() {}

func (x *WriteResponse) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[7]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WriteResponse.ProtoReflect.Descriptor instead.
func (*WriteResponse) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{7}
}

// Request to write HTML to a HTML writer.
type WriteHTMLRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Writer *HTMLWriter `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
	Html   string      `protobuf:"bytes,2,opt,name=html,proto3" json:"html,omitempty"`
}

func (x *WriteHTMLRequest) Reset() {
	*x = WriteHTMLRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[8]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteHTMLRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteHTMLRequest) ProtoMessage() {}

func (x *WriteHTMLRequest) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[8]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WriteHTMLRequest.ProtoReflect.Descriptor instead.
func (*WriteHTMLRequest) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{8}
}

func (x *WriteHTMLRequest) GetWriter() *HTMLWriter {
	if x != nil {
		return x.Writer
	}
	return nil
}

func (x *WriteHTMLRequest) GetHtml() string {
	if x != nil {
		return x.Html
	}
	return ""
}

type WriteHTMLResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *WriteHTMLResponse) Reset() {
	*x = WriteHTMLResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[9]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteHTMLResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteHTMLResponse) ProtoMessage() {}

func (x *WriteHTMLResponse) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[9]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WriteHTMLResponse.ProtoReflect.Descriptor instead.
func (*WriteHTMLResponse) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{9}
}

// Request to write CSS to a HTML writer.
type WriteCSSRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Writer *HTMLWriter `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
	Css    string      `protobuf:"bytes,2,opt,name=css,proto3" json:"css,omitempty"`
}

func (x *WriteCSSRequest) Reset() {
	*x = WriteCSSRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[10]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteCSSRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteCSSRequest) ProtoMessage() {}

func (x *WriteCSSRequest) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[10]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WriteCSSRequest.ProtoReflect.Descriptor instead.
func (*WriteCSSRequest) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{10}
}

func (x *WriteCSSRequest) GetWriter() *HTMLWriter {
	if x != nil {
		return x.Writer
	}
	return nil
}

func (x *WriteCSSRequest) GetCss() string {
	if x != nil {
		return x.Css
	}
	return ""
}

type WriteCSSResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *WriteCSSResponse) Reset() {
	*x = WriteCSSResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_text_proto_msgTypes[11]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteCSSResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteCSSResponse) ProtoMessage() {}

func (x *WriteCSSResponse) ProtoReflect() protoreflect.Message {
	mi := &file_text_proto_msgTypes[11]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WriteCSSResponse.ProtoReflect.Descriptor instead.
func (*WriteCSSResponse) Descriptor() ([]byte, []int) {
	return file_text_proto_rawDescGZIP(), []int{11}
}

var File_text_proto protoreflect.FileDescriptor

var file_text_proto_rawDesc = []byte{
	0x0a, 0x0a, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0f, 0x6d, 0x75,
	0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x1a, 0x0a, 0x74,
	0x72, 0x65, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0x5e, 0x0a, 0x06, 0x57, 0x72, 0x69,
	0x74, 0x65, 0x72, 0x12, 0x2a, 0x0a, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49, 0x44, 0x18, 0x01, 0x20,
	0x01, 0x28, 0x0b, 0x32, 0x12, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65,
	0x2e, 0x54, 0x72, 0x65, 0x65, 0x49, 0x44, 0x52, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49, 0x44, 0x12,
	0x28, 0x0a, 0x04, 0x70, 0x61, 0x74, 0x68, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e,
	0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x50,
	0x61, 0x74, 0x68, 0x52, 0x04, 0x70, 0x61, 0x74, 0x68, 0x22, 0x62, 0x0a, 0x0a, 0x48, 0x54, 0x4d,
	0x4c, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x2a, 0x0a, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49,
	0x44, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x12, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73,
	0x63, 0x6f, 0x70, 0x65, 0x2e, 0x54, 0x72, 0x65, 0x65, 0x49, 0x44, 0x52, 0x06, 0x74, 0x72, 0x65,
	0x65, 0x49, 0x44, 0x12, 0x28, 0x0a, 0x04, 0x70, 0x61, 0x74, 0x68, 0x18, 0x02, 0x20, 0x01, 0x28,
	0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x4e,
	0x6f, 0x64, 0x65, 0x50, 0x61, 0x74, 0x68, 0x52, 0x04, 0x70, 0x61, 0x74, 0x68, 0x22, 0x68, 0x0a,
	0x10, 0x4e, 0x65, 0x77, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73,
	0x74, 0x12, 0x2a, 0x0a, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49, 0x44, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x0b, 0x32, 0x12, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x54,
	0x72, 0x65, 0x65, 0x49, 0x44, 0x52, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49, 0x44, 0x12, 0x28, 0x0a,
	0x04, 0x70, 0x61, 0x74, 0x68, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75,
	0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x50, 0x61, 0x74,
	0x68, 0x52, 0x04, 0x70, 0x61, 0x74, 0x68, 0x22, 0x44, 0x0a, 0x11, 0x4e, 0x65, 0x77, 0x57, 0x72,
	0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x2f, 0x0a, 0x06,
	0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x6d,
	0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x22, 0x6c, 0x0a,
	0x14, 0x4e, 0x65, 0x77, 0x48, 0x54, 0x4d, 0x4c, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x2a, 0x0a, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49, 0x44, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x12, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f,
	0x70, 0x65, 0x2e, 0x54, 0x72, 0x65, 0x65, 0x49, 0x44, 0x52, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49,
	0x44, 0x12, 0x28, 0x0a, 0x04, 0x70, 0x61, 0x74, 0x68, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32,
	0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x4e, 0x6f, 0x64,
	0x65, 0x50, 0x61, 0x74, 0x68, 0x52, 0x04, 0x70, 0x61, 0x74, 0x68, 0x22, 0x4c, 0x0a, 0x15, 0x4e,
	0x65, 0x77, 0x48, 0x54, 0x4d, 0x4c, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70,
	0x6f, 0x6e, 0x73, 0x65, 0x12, 0x33, 0x0a, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x0b, 0x32, 0x1b, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70,
	0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x48, 0x54, 0x4d, 0x4c, 0x57, 0x72, 0x69, 0x74, 0x65,
	0x72, 0x52, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x22, 0x53, 0x0a, 0x0c, 0x57, 0x72, 0x69,
	0x74, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x2f, 0x0a, 0x06, 0x77, 0x72, 0x69,
	0x74, 0x65, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x6d, 0x75, 0x6c, 0x74,
	0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x57, 0x72, 0x69, 0x74,
	0x65, 0x72, 0x52, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x12, 0x0a, 0x04, 0x74, 0x65,
	0x78, 0x74, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x74, 0x65, 0x78, 0x74, 0x22, 0x0f,
	0x0a, 0x0d, 0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22,
	0x5b, 0x0a, 0x10, 0x57, 0x72, 0x69, 0x74, 0x65, 0x48, 0x54, 0x4d, 0x4c, 0x52, 0x65, 0x71, 0x75,
	0x65, 0x73, 0x74, 0x12, 0x33, 0x0a, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01, 0x20,
	0x01, 0x28, 0x0b, 0x32, 0x1b, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65,
	0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x48, 0x54, 0x4d, 0x4c, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72,
	0x52, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x12, 0x0a, 0x04, 0x68, 0x74, 0x6d, 0x6c,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x68, 0x74, 0x6d, 0x6c, 0x22, 0x13, 0x0a, 0x11,
	0x57, 0x72, 0x69, 0x74, 0x65, 0x48, 0x54, 0x4d, 0x4c, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73,
	0x65, 0x22, 0x58, 0x0a, 0x0f, 0x57, 0x72, 0x69, 0x74, 0x65, 0x43, 0x53, 0x53, 0x52, 0x65, 0x71,
	0x75, 0x65, 0x73, 0x74, 0x12, 0x33, 0x0a, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x0b, 0x32, 0x1b, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70,
	0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x48, 0x54, 0x4d, 0x4c, 0x57, 0x72, 0x69, 0x74, 0x65,
	0x72, 0x52, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x10, 0x0a, 0x03, 0x63, 0x73, 0x73,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x63, 0x73, 0x73, 0x22, 0x12, 0x0a, 0x10, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x43, 0x53, 0x53, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x32,
	0xb1, 0x03, 0x0a, 0x04, 0x54, 0x65, 0x78, 0x74, 0x12, 0x54, 0x0a, 0x09, 0x4e, 0x65, 0x77, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x21, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f,
	0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x4e, 0x65, 0x77, 0x57, 0x72, 0x69, 0x74, 0x65,
	0x72, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x22, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69,
	0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x4e, 0x65, 0x77, 0x57, 0x72,
	0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12, 0x60,
	0x0a, 0x0d, 0x4e, 0x65, 0x77, 0x48, 0x54, 0x4d, 0x4c, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12,
	0x25, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78,
	0x74, 0x2e, 0x4e, 0x65, 0x77, 0x48, 0x54, 0x4d, 0x4c, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x26, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63,
	0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x4e, 0x65, 0x77, 0x48, 0x54, 0x4d, 0x4c,
	0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00,
	0x12, 0x48, 0x0a, 0x05, 0x57, 0x72, 0x69, 0x74, 0x65, 0x12, 0x1d, 0x2e, 0x6d, 0x75, 0x6c, 0x74,
	0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x57, 0x72, 0x69, 0x74,
	0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1e, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69,
	0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65,
	0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12, 0x54, 0x0a, 0x09, 0x57, 0x72,
	0x69, 0x74, 0x65, 0x48, 0x54, 0x4d, 0x4c, 0x12, 0x21, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73,
	0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x48,
	0x54, 0x4d, 0x4c, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x22, 0x2e, 0x6d, 0x75, 0x6c,
	0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x57, 0x72, 0x69,
	0x74, 0x65, 0x48, 0x54, 0x4d, 0x4c, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00,
	0x12, 0x51, 0x0a, 0x08, 0x57, 0x72, 0x69, 0x74, 0x65, 0x43, 0x53, 0x53, 0x12, 0x20, 0x2e, 0x6d,
	0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74, 0x2e, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x43, 0x53, 0x53, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x21,
	0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x78, 0x74,
	0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x43, 0x53, 0x53, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73,
	0x65, 0x22, 0x00, 0x42, 0x21, 0x5a, 0x1f, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70,
	0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x74, 0x65, 0x78, 0x74, 0x5f, 0x67, 0x6f,
	0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_text_proto_rawDescOnce sync.Once
	file_text_proto_rawDescData = file_text_proto_rawDesc
)

func file_text_proto_rawDescGZIP() []byte {
	file_text_proto_rawDescOnce.Do(func() {
		file_text_proto_rawDescData = protoimpl.X.CompressGZIP(file_text_proto_rawDescData)
	})
	return file_text_proto_rawDescData
}

var file_text_proto_msgTypes = make([]protoimpl.MessageInfo, 12)
var file_text_proto_goTypes = []interface{}{
	(*Writer)(nil),                 // 0: multiscope.text.Writer
	(*HTMLWriter)(nil),             // 1: multiscope.text.HTMLWriter
	(*NewWriterRequest)(nil),       // 2: multiscope.text.NewWriterRequest
	(*NewWriterResponse)(nil),      // 3: multiscope.text.NewWriterResponse
	(*NewHTMLWriterRequest)(nil),   // 4: multiscope.text.NewHTMLWriterRequest
	(*NewHTMLWriterResponse)(nil),  // 5: multiscope.text.NewHTMLWriterResponse
	(*WriteRequest)(nil),           // 6: multiscope.text.WriteRequest
	(*WriteResponse)(nil),          // 7: multiscope.text.WriteResponse
	(*WriteHTMLRequest)(nil),       // 8: multiscope.text.WriteHTMLRequest
	(*WriteHTMLResponse)(nil),      // 9: multiscope.text.WriteHTMLResponse
	(*WriteCSSRequest)(nil),        // 10: multiscope.text.WriteCSSRequest
	(*WriteCSSResponse)(nil),       // 11: multiscope.text.WriteCSSResponse
	(*tree_go_proto.TreeID)(nil),   // 12: multiscope.TreeID
	(*tree_go_proto.NodePath)(nil), // 13: multiscope.NodePath
}
var file_text_proto_depIdxs = []int32{
	12, // 0: multiscope.text.Writer.treeID:type_name -> multiscope.TreeID
	13, // 1: multiscope.text.Writer.path:type_name -> multiscope.NodePath
	12, // 2: multiscope.text.HTMLWriter.treeID:type_name -> multiscope.TreeID
	13, // 3: multiscope.text.HTMLWriter.path:type_name -> multiscope.NodePath
	12, // 4: multiscope.text.NewWriterRequest.treeID:type_name -> multiscope.TreeID
	13, // 5: multiscope.text.NewWriterRequest.path:type_name -> multiscope.NodePath
	0,  // 6: multiscope.text.NewWriterResponse.writer:type_name -> multiscope.text.Writer
	12, // 7: multiscope.text.NewHTMLWriterRequest.treeID:type_name -> multiscope.TreeID
	13, // 8: multiscope.text.NewHTMLWriterRequest.path:type_name -> multiscope.NodePath
	1,  // 9: multiscope.text.NewHTMLWriterResponse.writer:type_name -> multiscope.text.HTMLWriter
	0,  // 10: multiscope.text.WriteRequest.writer:type_name -> multiscope.text.Writer
	1,  // 11: multiscope.text.WriteHTMLRequest.writer:type_name -> multiscope.text.HTMLWriter
	1,  // 12: multiscope.text.WriteCSSRequest.writer:type_name -> multiscope.text.HTMLWriter
	2,  // 13: multiscope.text.Text.NewWriter:input_type -> multiscope.text.NewWriterRequest
	4,  // 14: multiscope.text.Text.NewHTMLWriter:input_type -> multiscope.text.NewHTMLWriterRequest
	6,  // 15: multiscope.text.Text.Write:input_type -> multiscope.text.WriteRequest
	8,  // 16: multiscope.text.Text.WriteHTML:input_type -> multiscope.text.WriteHTMLRequest
	10, // 17: multiscope.text.Text.WriteCSS:input_type -> multiscope.text.WriteCSSRequest
	3,  // 18: multiscope.text.Text.NewWriter:output_type -> multiscope.text.NewWriterResponse
	5,  // 19: multiscope.text.Text.NewHTMLWriter:output_type -> multiscope.text.NewHTMLWriterResponse
	7,  // 20: multiscope.text.Text.Write:output_type -> multiscope.text.WriteResponse
	9,  // 21: multiscope.text.Text.WriteHTML:output_type -> multiscope.text.WriteHTMLResponse
	11, // 22: multiscope.text.Text.WriteCSS:output_type -> multiscope.text.WriteCSSResponse
	18, // [18:23] is the sub-list for method output_type
	13, // [13:18] is the sub-list for method input_type
	13, // [13:13] is the sub-list for extension type_name
	13, // [13:13] is the sub-list for extension extendee
	0,  // [0:13] is the sub-list for field type_name
}

func init() { file_text_proto_init() }
func file_text_proto_init() {
	if File_text_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_text_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Writer); i {
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
		file_text_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*HTMLWriter); i {
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
		file_text_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NewWriterRequest); i {
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
		file_text_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NewWriterResponse); i {
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
		file_text_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NewHTMLWriterRequest); i {
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
		file_text_proto_msgTypes[5].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NewHTMLWriterResponse); i {
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
		file_text_proto_msgTypes[6].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WriteRequest); i {
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
		file_text_proto_msgTypes[7].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WriteResponse); i {
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
		file_text_proto_msgTypes[8].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WriteHTMLRequest); i {
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
		file_text_proto_msgTypes[9].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WriteHTMLResponse); i {
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
		file_text_proto_msgTypes[10].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WriteCSSRequest); i {
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
		file_text_proto_msgTypes[11].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WriteCSSResponse); i {
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
			RawDescriptor: file_text_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   12,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_text_proto_goTypes,
		DependencyIndexes: file_text_proto_depIdxs,
		MessageInfos:      file_text_proto_msgTypes,
	}.Build()
	File_text_proto = out.File
	file_text_proto_rawDesc = nil
	file_text_proto_goTypes = nil
	file_text_proto_depIdxs = nil
}
