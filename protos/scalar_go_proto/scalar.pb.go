// Protocol buffer to stream text data.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.30.0
// 	protoc        v3.21.12
// source: scalar.proto

package scalar_go_proto

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
		mi := &file_scalar_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Writer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Writer) ProtoMessage() {}

func (x *Writer) ProtoReflect() protoreflect.Message {
	mi := &file_scalar_proto_msgTypes[0]
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
	return file_scalar_proto_rawDescGZIP(), []int{0}
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

// Request to create a new writer in the tree.
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
		mi := &file_scalar_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewWriterRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewWriterRequest) ProtoMessage() {}

func (x *NewWriterRequest) ProtoReflect() protoreflect.Message {
	mi := &file_scalar_proto_msgTypes[1]
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
	return file_scalar_proto_rawDescGZIP(), []int{1}
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

// Response after creating a new writer in the tree.
type NewWriterResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Writer *Writer `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
}

func (x *NewWriterResponse) Reset() {
	*x = NewWriterResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_scalar_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewWriterResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewWriterResponse) ProtoMessage() {}

func (x *NewWriterResponse) ProtoReflect() protoreflect.Message {
	mi := &file_scalar_proto_msgTypes[2]
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
	return file_scalar_proto_rawDescGZIP(), []int{2}
}

func (x *NewWriterResponse) GetWriter() *Writer {
	if x != nil {
		return x.Writer
	}
	return nil
}

// Request to write a scalars data.
type WriteRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Writer       *Writer            `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
	LabelToValue map[string]float64 `protobuf:"bytes,2,rep,name=label_to_value,json=labelToValue,proto3" json:"label_to_value,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"fixed64,2,opt,name=value,proto3"`
}

func (x *WriteRequest) Reset() {
	*x = WriteRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_scalar_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteRequest) ProtoMessage() {}

func (x *WriteRequest) ProtoReflect() protoreflect.Message {
	mi := &file_scalar_proto_msgTypes[3]
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
	return file_scalar_proto_rawDescGZIP(), []int{3}
}

func (x *WriteRequest) GetWriter() *Writer {
	if x != nil {
		return x.Writer
	}
	return nil
}

func (x *WriteRequest) GetLabelToValue() map[string]float64 {
	if x != nil {
		return x.LabelToValue
	}
	return nil
}

type WriteResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *WriteResponse) Reset() {
	*x = WriteResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_scalar_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteResponse) ProtoMessage() {}

func (x *WriteResponse) ProtoReflect() protoreflect.Message {
	mi := &file_scalar_proto_msgTypes[4]
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
	return file_scalar_proto_rawDescGZIP(), []int{4}
}

var File_scalar_proto protoreflect.FileDescriptor

var file_scalar_proto_rawDesc = []byte{
	0x0a, 0x0c, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x11,
	0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x73, 0x63, 0x61, 0x6c, 0x61,
	0x72, 0x1a, 0x0a, 0x74, 0x72, 0x65, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0x5e, 0x0a,
	0x06, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x2a, 0x0a, 0x06, 0x74, 0x72, 0x65, 0x65, 0x49,
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
	0x68, 0x52, 0x04, 0x70, 0x61, 0x74, 0x68, 0x22, 0x46, 0x0a, 0x11, 0x4e, 0x65, 0x77, 0x57, 0x72,
	0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x31, 0x0a, 0x06,
	0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x19, 0x2e, 0x6d,
	0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72,
	0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x22,
	0xdb, 0x01, 0x0a, 0x0c, 0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74,
	0x12, 0x31, 0x0a, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b,
	0x32, 0x19, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x73, 0x63,
	0x61, 0x6c, 0x61, 0x72, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x06, 0x77, 0x72, 0x69,
	0x74, 0x65, 0x72, 0x12, 0x57, 0x0a, 0x0e, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x5f, 0x74, 0x6f, 0x5f,
	0x76, 0x61, 0x6c, 0x75, 0x65, 0x18, 0x02, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x31, 0x2e, 0x6d, 0x75,
	0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x2e,
	0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x2e, 0x4c, 0x61, 0x62,
	0x65, 0x6c, 0x54, 0x6f, 0x56, 0x61, 0x6c, 0x75, 0x65, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x52, 0x0c,
	0x6c, 0x61, 0x62, 0x65, 0x6c, 0x54, 0x6f, 0x56, 0x61, 0x6c, 0x75, 0x65, 0x1a, 0x3f, 0x0a, 0x11,
	0x4c, 0x61, 0x62, 0x65, 0x6c, 0x54, 0x6f, 0x56, 0x61, 0x6c, 0x75, 0x65, 0x45, 0x6e, 0x74, 0x72,
	0x79, 0x12, 0x10, 0x0a, 0x03, 0x6b, 0x65, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03,
	0x6b, 0x65, 0x79, 0x12, 0x14, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x18, 0x02, 0x20, 0x01,
	0x28, 0x01, 0x52, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x3a, 0x02, 0x38, 0x01, 0x22, 0x0f, 0x0a,
	0x0d, 0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x32, 0xb1,
	0x01, 0x0a, 0x07, 0x53, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x73, 0x12, 0x58, 0x0a, 0x09, 0x4e, 0x65,
	0x77, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x23, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73,
	0x63, 0x6f, 0x70, 0x65, 0x2e, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x2e, 0x4e, 0x65, 0x77, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x24, 0x2e, 0x6d,
	0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72,
	0x2e, 0x4e, 0x65, 0x77, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e,
	0x73, 0x65, 0x22, 0x00, 0x12, 0x4c, 0x0a, 0x05, 0x57, 0x72, 0x69, 0x74, 0x65, 0x12, 0x1f, 0x2e,
	0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x73, 0x63, 0x61, 0x6c, 0x61,
	0x72, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x20,
	0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x73, 0x63, 0x61, 0x6c,
	0x61, 0x72, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65,
	0x22, 0x00, 0x42, 0x23, 0x5a, 0x21, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65,
	0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x5f, 0x67,
	0x6f, 0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_scalar_proto_rawDescOnce sync.Once
	file_scalar_proto_rawDescData = file_scalar_proto_rawDesc
)

func file_scalar_proto_rawDescGZIP() []byte {
	file_scalar_proto_rawDescOnce.Do(func() {
		file_scalar_proto_rawDescData = protoimpl.X.CompressGZIP(file_scalar_proto_rawDescData)
	})
	return file_scalar_proto_rawDescData
}

var file_scalar_proto_msgTypes = make([]protoimpl.MessageInfo, 6)
var file_scalar_proto_goTypes = []interface{}{
	(*Writer)(nil),                 // 0: multiscope.scalar.Writer
	(*NewWriterRequest)(nil),       // 1: multiscope.scalar.NewWriterRequest
	(*NewWriterResponse)(nil),      // 2: multiscope.scalar.NewWriterResponse
	(*WriteRequest)(nil),           // 3: multiscope.scalar.WriteRequest
	(*WriteResponse)(nil),          // 4: multiscope.scalar.WriteResponse
	nil,                            // 5: multiscope.scalar.WriteRequest.LabelToValueEntry
	(*tree_go_proto.TreeID)(nil),   // 6: multiscope.TreeID
	(*tree_go_proto.NodePath)(nil), // 7: multiscope.NodePath
}
var file_scalar_proto_depIdxs = []int32{
	6, // 0: multiscope.scalar.Writer.treeID:type_name -> multiscope.TreeID
	7, // 1: multiscope.scalar.Writer.path:type_name -> multiscope.NodePath
	6, // 2: multiscope.scalar.NewWriterRequest.treeID:type_name -> multiscope.TreeID
	7, // 3: multiscope.scalar.NewWriterRequest.path:type_name -> multiscope.NodePath
	0, // 4: multiscope.scalar.NewWriterResponse.writer:type_name -> multiscope.scalar.Writer
	0, // 5: multiscope.scalar.WriteRequest.writer:type_name -> multiscope.scalar.Writer
	5, // 6: multiscope.scalar.WriteRequest.label_to_value:type_name -> multiscope.scalar.WriteRequest.LabelToValueEntry
	1, // 7: multiscope.scalar.Scalars.NewWriter:input_type -> multiscope.scalar.NewWriterRequest
	3, // 8: multiscope.scalar.Scalars.Write:input_type -> multiscope.scalar.WriteRequest
	2, // 9: multiscope.scalar.Scalars.NewWriter:output_type -> multiscope.scalar.NewWriterResponse
	4, // 10: multiscope.scalar.Scalars.Write:output_type -> multiscope.scalar.WriteResponse
	9, // [9:11] is the sub-list for method output_type
	7, // [7:9] is the sub-list for method input_type
	7, // [7:7] is the sub-list for extension type_name
	7, // [7:7] is the sub-list for extension extendee
	0, // [0:7] is the sub-list for field type_name
}

func init() { file_scalar_proto_init() }
func file_scalar_proto_init() {
	if File_scalar_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_scalar_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
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
		file_scalar_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
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
		file_scalar_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
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
		file_scalar_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
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
		file_scalar_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
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
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_scalar_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   6,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_scalar_proto_goTypes,
		DependencyIndexes: file_scalar_proto_depIdxs,
		MessageInfos:      file_scalar_proto_msgTypes,
	}.Build()
	File_scalar_proto = out.File
	file_scalar_proto_rawDesc = nil
	file_scalar_proto_goTypes = nil
	file_scalar_proto_depIdxs = nil
}
