// Protocol buffer to stream tensors.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.0
// 	protoc        v3.19.4
// source: tensor.proto

package tensor_go_proto

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

type DataType int32

const (
	DataType_DT_INVALID DataType = 0
	DataType_DT_FLOAT32 DataType = 1
	DataType_DT_UINT8   DataType = 4
)

// Enum value maps for DataType.
var (
	DataType_name = map[int32]string{
		0: "DT_INVALID",
		1: "DT_FLOAT32",
		4: "DT_UINT8",
	}
	DataType_value = map[string]int32{
		"DT_INVALID": 0,
		"DT_FLOAT32": 1,
		"DT_UINT8":   4,
	}
)

func (x DataType) Enum() *DataType {
	p := new(DataType)
	*p = x
	return p
}

func (x DataType) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (DataType) Descriptor() protoreflect.EnumDescriptor {
	return file_tensor_proto_enumTypes[0].Descriptor()
}

func (DataType) Type() protoreflect.EnumType {
	return &file_tensor_proto_enumTypes[0]
}

func (x DataType) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use DataType.Descriptor instead.
func (DataType) EnumDescriptor() ([]byte, []int) {
	return file_tensor_proto_rawDescGZIP(), []int{0}
}

type Writer struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Path in the Multiscope tree.
	Path *tree_go_proto.NodePath `protobuf:"bytes,1,opt,name=path,proto3" json:"path,omitempty"`
}

func (x *Writer) Reset() {
	*x = Writer{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensor_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Writer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Writer) ProtoMessage() {}

func (x *Writer) ProtoReflect() protoreflect.Message {
	mi := &file_tensor_proto_msgTypes[0]
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
	return file_tensor_proto_rawDescGZIP(), []int{0}
}

func (x *Writer) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

// Dimensions of a tensor.
type Shape struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Dimensions of the tensor, such as {"input", 30}, {"output", 40}
	// for a 30 x 40 2D tensor.  If an entry has size -1, this
	// corresponds to a dimension of unknown size. The names are
	// optional.
	//
	// The order of entries in "dim" matters: It indicates the layout of the
	// values in the tensor in-memory representation.
	//
	// The first entry in "dim" is the outermost dimension used to layout the
	// values, the last entry is the innermost dimension.  This matches the
	// in-memory layout of RowMajor Eigen tensors.
	Dim []*Shape_Dim `protobuf:"bytes,2,rep,name=dim,proto3" json:"dim,omitempty"`
}

func (x *Shape) Reset() {
	*x = Shape{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensor_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Shape) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Shape) ProtoMessage() {}

func (x *Shape) ProtoReflect() protoreflect.Message {
	mi := &file_tensor_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Shape.ProtoReflect.Descriptor instead.
func (*Shape) Descriptor() ([]byte, []int) {
	return file_tensor_proto_rawDescGZIP(), []int{1}
}

func (x *Shape) GetDim() []*Shape_Dim {
	if x != nil {
		return x.Dim
	}
	return nil
}

type Tensor struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Dtype DataType `protobuf:"varint,1,opt,name=dtype,proto3,enum=multiscope.tensors.DataType" json:"dtype,omitempty"`
	// Shape of the tensor.  TODO(mdevin): sort out the 0-rank issues.
	Shape *Shape `protobuf:"bytes,2,opt,name=shape,proto3" json:"shape,omitempty"`
	// Serialized raw tensor content from either Tensor::AsProtoTensorContent or
	// memcpy in tensorflow::grpc::EncodeTensorToByteBuffer. This representation
	// can be used for all tensor types. The purpose of this representation is to
	// reduce serialization overhead during RPC call by avoiding serialization of
	// many repeated small items.
	Content []byte `protobuf:"bytes,4,opt,name=content,proto3" json:"content,omitempty"`
}

func (x *Tensor) Reset() {
	*x = Tensor{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensor_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Tensor) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Tensor) ProtoMessage() {}

func (x *Tensor) ProtoReflect() protoreflect.Message {
	mi := &file_tensor_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Tensor.ProtoReflect.Descriptor instead.
func (*Tensor) Descriptor() ([]byte, []int) {
	return file_tensor_proto_rawDescGZIP(), []int{2}
}

func (x *Tensor) GetDtype() DataType {
	if x != nil {
		return x.Dtype
	}
	return DataType_DT_INVALID
}

func (x *Tensor) GetShape() *Shape {
	if x != nil {
		return x.Shape
	}
	return nil
}

func (x *Tensor) GetContent() []byte {
	if x != nil {
		return x.Content
	}
	return nil
}

// Request to create a new TensorWriter in the tree.
type NewWriterRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Path *tree_go_proto.NodePath `protobuf:"bytes,1,opt,name=path,proto3" json:"path,omitempty"`
}

func (x *NewWriterRequest) Reset() {
	*x = NewWriterRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensor_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewWriterRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewWriterRequest) ProtoMessage() {}

func (x *NewWriterRequest) ProtoReflect() protoreflect.Message {
	mi := &file_tensor_proto_msgTypes[3]
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
	return file_tensor_proto_rawDescGZIP(), []int{3}
}

func (x *NewWriterRequest) GetPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.Path
	}
	return nil
}

// Response after creating a new TensorWriter in the tree.
type NewWriterResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Writer           *Writer                 `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
	DefaultPanelPath *tree_go_proto.NodePath `protobuf:"bytes,2,opt,name=defaultPanelPath,proto3" json:"defaultPanelPath,omitempty"`
}

func (x *NewWriterResponse) Reset() {
	*x = NewWriterResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensor_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NewWriterResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NewWriterResponse) ProtoMessage() {}

func (x *NewWriterResponse) ProtoReflect() protoreflect.Message {
	mi := &file_tensor_proto_msgTypes[4]
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
	return file_tensor_proto_rawDescGZIP(), []int{4}
}

func (x *NewWriterResponse) GetWriter() *Writer {
	if x != nil {
		return x.Writer
	}
	return nil
}

func (x *NewWriterResponse) GetDefaultPanelPath() *tree_go_proto.NodePath {
	if x != nil {
		return x.DefaultPanelPath
	}
	return nil
}

// Request to write a tensor.
type WriteRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Writer to write the data to.
	Writer *Writer `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
	// Data to write.
	Tensor *Tensor `protobuf:"bytes,2,opt,name=tensor,proto3" json:"tensor,omitempty"`
}

func (x *WriteRequest) Reset() {
	*x = WriteRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensor_proto_msgTypes[5]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteRequest) ProtoMessage() {}

func (x *WriteRequest) ProtoReflect() protoreflect.Message {
	mi := &file_tensor_proto_msgTypes[5]
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
	return file_tensor_proto_rawDescGZIP(), []int{5}
}

func (x *WriteRequest) GetWriter() *Writer {
	if x != nil {
		return x.Writer
	}
	return nil
}

func (x *WriteRequest) GetTensor() *Tensor {
	if x != nil {
		return x.Tensor
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
		mi := &file_tensor_proto_msgTypes[6]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WriteResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WriteResponse) ProtoMessage() {}

func (x *WriteResponse) ProtoReflect() protoreflect.Message {
	mi := &file_tensor_proto_msgTypes[6]
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
	return file_tensor_proto_rawDescGZIP(), []int{6}
}

type ResetWriterRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Writer to reset.
	Writer *Writer `protobuf:"bytes,1,opt,name=writer,proto3" json:"writer,omitempty"`
}

func (x *ResetWriterRequest) Reset() {
	*x = ResetWriterRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensor_proto_msgTypes[7]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ResetWriterRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResetWriterRequest) ProtoMessage() {}

func (x *ResetWriterRequest) ProtoReflect() protoreflect.Message {
	mi := &file_tensor_proto_msgTypes[7]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ResetWriterRequest.ProtoReflect.Descriptor instead.
func (*ResetWriterRequest) Descriptor() ([]byte, []int) {
	return file_tensor_proto_rawDescGZIP(), []int{7}
}

func (x *ResetWriterRequest) GetWriter() *Writer {
	if x != nil {
		return x.Writer
	}
	return nil
}

type ResetWriterResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *ResetWriterResponse) Reset() {
	*x = ResetWriterResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensor_proto_msgTypes[8]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ResetWriterResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResetWriterResponse) ProtoMessage() {}

func (x *ResetWriterResponse) ProtoReflect() protoreflect.Message {
	mi := &file_tensor_proto_msgTypes[8]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ResetWriterResponse.ProtoReflect.Descriptor instead.
func (*ResetWriterResponse) Descriptor() ([]byte, []int) {
	return file_tensor_proto_rawDescGZIP(), []int{8}
}

// One dimension of the tensor.
type Shape_Dim struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Size of the tensor in that dimension.
	Size int64 `protobuf:"varint,1,opt,name=size,proto3" json:"size,omitempty"`
	// Optional name of the tensor dimension.
	Name string `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`
}

func (x *Shape_Dim) Reset() {
	*x = Shape_Dim{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensor_proto_msgTypes[9]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Shape_Dim) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Shape_Dim) ProtoMessage() {}

func (x *Shape_Dim) ProtoReflect() protoreflect.Message {
	mi := &file_tensor_proto_msgTypes[9]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Shape_Dim.ProtoReflect.Descriptor instead.
func (*Shape_Dim) Descriptor() ([]byte, []int) {
	return file_tensor_proto_rawDescGZIP(), []int{1, 0}
}

func (x *Shape_Dim) GetSize() int64 {
	if x != nil {
		return x.Size
	}
	return 0
}

func (x *Shape_Dim) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

var File_tensor_proto protoreflect.FileDescriptor

var file_tensor_proto_rawDesc = []byte{
	0x0a, 0x0c, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x12,
	0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f,
	0x72, 0x73, 0x1a, 0x0a, 0x74, 0x72, 0x65, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0x32,
	0x0a, 0x06, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x28, 0x0a, 0x04, 0x70, 0x61, 0x74, 0x68,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63,
	0x6f, 0x70, 0x65, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x50, 0x61, 0x74, 0x68, 0x52, 0x04, 0x70, 0x61,
	0x74, 0x68, 0x22, 0x67, 0x0a, 0x05, 0x53, 0x68, 0x61, 0x70, 0x65, 0x12, 0x2f, 0x0a, 0x03, 0x64,
	0x69, 0x6d, 0x18, 0x02, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x1d, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69,
	0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x2e, 0x53, 0x68,
	0x61, 0x70, 0x65, 0x2e, 0x44, 0x69, 0x6d, 0x52, 0x03, 0x64, 0x69, 0x6d, 0x1a, 0x2d, 0x0a, 0x03,
	0x44, 0x69, 0x6d, 0x12, 0x12, 0x0a, 0x04, 0x73, 0x69, 0x7a, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x03, 0x52, 0x04, 0x73, 0x69, 0x7a, 0x65, 0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x22, 0x8b, 0x01, 0x0a, 0x06,
	0x54, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x12, 0x32, 0x0a, 0x05, 0x64, 0x74, 0x79, 0x70, 0x65, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x0e, 0x32, 0x1c, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f,
	0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x2e, 0x44, 0x61, 0x74, 0x61, 0x54,
	0x79, 0x70, 0x65, 0x52, 0x05, 0x64, 0x74, 0x79, 0x70, 0x65, 0x12, 0x2f, 0x0a, 0x05, 0x73, 0x68,
	0x61, 0x70, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x19, 0x2e, 0x6d, 0x75, 0x6c, 0x74,
	0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x2e, 0x53,
	0x68, 0x61, 0x70, 0x65, 0x52, 0x05, 0x73, 0x68, 0x61, 0x70, 0x65, 0x12, 0x1c, 0x0a, 0x07, 0x63,
	0x6f, 0x6e, 0x74, 0x65, 0x6e, 0x74, 0x18, 0x04, 0x20, 0x01, 0x28, 0x0c, 0x42, 0x02, 0x08, 0x01,
	0x52, 0x07, 0x63, 0x6f, 0x6e, 0x74, 0x65, 0x6e, 0x74, 0x22, 0x3c, 0x0a, 0x10, 0x4e, 0x65, 0x77,
	0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x28, 0x0a,
	0x04, 0x70, 0x61, 0x74, 0x68, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75,
	0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x50, 0x61, 0x74,
	0x68, 0x52, 0x04, 0x70, 0x61, 0x74, 0x68, 0x22, 0x89, 0x01, 0x0a, 0x11, 0x4e, 0x65, 0x77, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x32, 0x0a,
	0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1a, 0x2e,
	0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f,
	0x72, 0x73, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65,
	0x72, 0x12, 0x40, 0x0a, 0x10, 0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x50, 0x61, 0x6e, 0x65,
	0x6c, 0x50, 0x61, 0x74, 0x68, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75,
	0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x50, 0x61, 0x74,
	0x68, 0x52, 0x10, 0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x50, 0x61, 0x6e, 0x65, 0x6c, 0x50,
	0x61, 0x74, 0x68, 0x22, 0x76, 0x0a, 0x0c, 0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x65, 0x71, 0x75,
	0x65, 0x73, 0x74, 0x12, 0x32, 0x0a, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01, 0x20,
	0x01, 0x28, 0x0b, 0x32, 0x1a, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65,
	0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52,
	0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x32, 0x0a, 0x06, 0x74, 0x65, 0x6e, 0x73, 0x6f,
	0x72, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1a, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73,
	0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x2e, 0x54, 0x65, 0x6e,
	0x73, 0x6f, 0x72, 0x52, 0x06, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x22, 0x0f, 0x0a, 0x0d, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x48, 0x0a, 0x12,
	0x52, 0x65, 0x73, 0x65, 0x74, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x71, 0x75, 0x65,
	0x73, 0x74, 0x12, 0x32, 0x0a, 0x06, 0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x18, 0x01, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x1a, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e,
	0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x06,
	0x77, 0x72, 0x69, 0x74, 0x65, 0x72, 0x22, 0x15, 0x0a, 0x13, 0x52, 0x65, 0x73, 0x65, 0x74, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x2a, 0x38, 0x0a,
	0x08, 0x44, 0x61, 0x74, 0x61, 0x54, 0x79, 0x70, 0x65, 0x12, 0x0e, 0x0a, 0x0a, 0x44, 0x54, 0x5f,
	0x49, 0x4e, 0x56, 0x41, 0x4c, 0x49, 0x44, 0x10, 0x00, 0x12, 0x0e, 0x0a, 0x0a, 0x44, 0x54, 0x5f,
	0x46, 0x4c, 0x4f, 0x41, 0x54, 0x33, 0x32, 0x10, 0x01, 0x12, 0x0c, 0x0a, 0x08, 0x44, 0x54, 0x5f,
	0x55, 0x49, 0x4e, 0x54, 0x38, 0x10, 0x04, 0x32, 0x97, 0x02, 0x0a, 0x07, 0x54, 0x65, 0x6e, 0x73,
	0x6f, 0x72, 0x73, 0x12, 0x5a, 0x0a, 0x09, 0x4e, 0x65, 0x77, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72,
	0x12, 0x24, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65,
	0x6e, 0x73, 0x6f, 0x72, 0x73, 0x2e, 0x4e, 0x65, 0x77, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x25, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63,
	0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x2e, 0x4e, 0x65, 0x77, 0x57,
	0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12,
	0x60, 0x0a, 0x0b, 0x52, 0x65, 0x73, 0x65, 0x74, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x12, 0x26,
	0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73,
	0x6f, 0x72, 0x73, 0x2e, 0x52, 0x65, 0x73, 0x65, 0x74, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x27, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63,
	0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x2e, 0x52, 0x65, 0x73, 0x65,
	0x74, 0x57, 0x72, 0x69, 0x74, 0x65, 0x72, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22,
	0x00, 0x12, 0x4e, 0x0a, 0x05, 0x57, 0x72, 0x69, 0x74, 0x65, 0x12, 0x20, 0x2e, 0x6d, 0x75, 0x6c,
	0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x2e,
	0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x21, 0x2e, 0x6d,
	0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72,
	0x73, 0x2e, 0x57, 0x72, 0x69, 0x74, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22,
	0x00, 0x42, 0x23, 0x5a, 0x21, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2f,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x5f, 0x67, 0x6f,
	0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_tensor_proto_rawDescOnce sync.Once
	file_tensor_proto_rawDescData = file_tensor_proto_rawDesc
)

func file_tensor_proto_rawDescGZIP() []byte {
	file_tensor_proto_rawDescOnce.Do(func() {
		file_tensor_proto_rawDescData = protoimpl.X.CompressGZIP(file_tensor_proto_rawDescData)
	})
	return file_tensor_proto_rawDescData
}

var file_tensor_proto_enumTypes = make([]protoimpl.EnumInfo, 1)
var file_tensor_proto_msgTypes = make([]protoimpl.MessageInfo, 10)
var file_tensor_proto_goTypes = []interface{}{
	(DataType)(0),                  // 0: multiscope.tensors.DataType
	(*Writer)(nil),                 // 1: multiscope.tensors.Writer
	(*Shape)(nil),                  // 2: multiscope.tensors.Shape
	(*Tensor)(nil),                 // 3: multiscope.tensors.Tensor
	(*NewWriterRequest)(nil),       // 4: multiscope.tensors.NewWriterRequest
	(*NewWriterResponse)(nil),      // 5: multiscope.tensors.NewWriterResponse
	(*WriteRequest)(nil),           // 6: multiscope.tensors.WriteRequest
	(*WriteResponse)(nil),          // 7: multiscope.tensors.WriteResponse
	(*ResetWriterRequest)(nil),     // 8: multiscope.tensors.ResetWriterRequest
	(*ResetWriterResponse)(nil),    // 9: multiscope.tensors.ResetWriterResponse
	(*Shape_Dim)(nil),              // 10: multiscope.tensors.Shape.Dim
	(*tree_go_proto.NodePath)(nil), // 11: multiscope.NodePath
}
var file_tensor_proto_depIdxs = []int32{
	11, // 0: multiscope.tensors.Writer.path:type_name -> multiscope.NodePath
	10, // 1: multiscope.tensors.Shape.dim:type_name -> multiscope.tensors.Shape.Dim
	0,  // 2: multiscope.tensors.Tensor.dtype:type_name -> multiscope.tensors.DataType
	2,  // 3: multiscope.tensors.Tensor.shape:type_name -> multiscope.tensors.Shape
	11, // 4: multiscope.tensors.NewWriterRequest.path:type_name -> multiscope.NodePath
	1,  // 5: multiscope.tensors.NewWriterResponse.writer:type_name -> multiscope.tensors.Writer
	11, // 6: multiscope.tensors.NewWriterResponse.defaultPanelPath:type_name -> multiscope.NodePath
	1,  // 7: multiscope.tensors.WriteRequest.writer:type_name -> multiscope.tensors.Writer
	3,  // 8: multiscope.tensors.WriteRequest.tensor:type_name -> multiscope.tensors.Tensor
	1,  // 9: multiscope.tensors.ResetWriterRequest.writer:type_name -> multiscope.tensors.Writer
	4,  // 10: multiscope.tensors.Tensors.NewWriter:input_type -> multiscope.tensors.NewWriterRequest
	8,  // 11: multiscope.tensors.Tensors.ResetWriter:input_type -> multiscope.tensors.ResetWriterRequest
	6,  // 12: multiscope.tensors.Tensors.Write:input_type -> multiscope.tensors.WriteRequest
	5,  // 13: multiscope.tensors.Tensors.NewWriter:output_type -> multiscope.tensors.NewWriterResponse
	9,  // 14: multiscope.tensors.Tensors.ResetWriter:output_type -> multiscope.tensors.ResetWriterResponse
	7,  // 15: multiscope.tensors.Tensors.Write:output_type -> multiscope.tensors.WriteResponse
	13, // [13:16] is the sub-list for method output_type
	10, // [10:13] is the sub-list for method input_type
	10, // [10:10] is the sub-list for extension type_name
	10, // [10:10] is the sub-list for extension extendee
	0,  // [0:10] is the sub-list for field type_name
}

func init() { file_tensor_proto_init() }
func file_tensor_proto_init() {
	if File_tensor_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_tensor_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
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
		file_tensor_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Shape); i {
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
		file_tensor_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Tensor); i {
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
		file_tensor_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
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
		file_tensor_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
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
		file_tensor_proto_msgTypes[5].Exporter = func(v interface{}, i int) interface{} {
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
		file_tensor_proto_msgTypes[6].Exporter = func(v interface{}, i int) interface{} {
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
		file_tensor_proto_msgTypes[7].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ResetWriterRequest); i {
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
		file_tensor_proto_msgTypes[8].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ResetWriterResponse); i {
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
		file_tensor_proto_msgTypes[9].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Shape_Dim); i {
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
			RawDescriptor: file_tensor_proto_rawDesc,
			NumEnums:      1,
			NumMessages:   10,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_tensor_proto_goTypes,
		DependencyIndexes: file_tensor_proto_depIdxs,
		EnumInfos:         file_tensor_proto_enumTypes,
		MessageInfos:      file_tensor_proto_msgTypes,
	}.Build()
	File_tensor_proto = out.File
	file_tensor_proto_rawDesc = nil
	file_tensor_proto_goTypes = nil
	file_tensor_proto_depIdxs = nil
}
