// Protocol buffer to stream scientific data.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.0
// 	protoc        v3.12.4
// source: ui.proto

package ui_go_proto

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

// Worker acknowledgement.
type WorkerAck struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *WorkerAck) Reset() {
	*x = WorkerAck{}
	if protoimpl.UnsafeEnabled {
		mi := &file_ui_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *WorkerAck) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*WorkerAck) ProtoMessage() {}

func (x *WorkerAck) ProtoReflect() protoreflect.Message {
	mi := &file_ui_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use WorkerAck.ProtoReflect.Descriptor instead.
func (*WorkerAck) Descriptor() ([]byte, []int) {
	return file_ui_proto_rawDescGZIP(), []int{0}
}

// Message to connect to the httpgrpc server.
type Connect struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Scheme string `protobuf:"bytes,1,opt,name=scheme,proto3" json:"scheme,omitempty"`
	Host   string `protobuf:"bytes,2,opt,name=host,proto3" json:"host,omitempty"`
}

func (x *Connect) Reset() {
	*x = Connect{}
	if protoimpl.UnsafeEnabled {
		mi := &file_ui_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Connect) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Connect) ProtoMessage() {}

func (x *Connect) ProtoReflect() protoreflect.Message {
	mi := &file_ui_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Connect.ProtoReflect.Descriptor instead.
func (*Connect) Descriptor() ([]byte, []int) {
	return file_ui_proto_rawDescGZIP(), []int{1}
}

func (x *Connect) GetScheme() string {
	if x != nil {
		return x.Scheme
	}
	return ""
}

func (x *Connect) GetHost() string {
	if x != nil {
		return x.Host
	}
	return ""
}

// Message to pull data from the server.
type Pull struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Actives []uint32 `protobuf:"varint,1,rep,packed,name=actives,proto3" json:"actives,omitempty"`
}

func (x *Pull) Reset() {
	*x = Pull{}
	if protoimpl.UnsafeEnabled {
		mi := &file_ui_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Pull) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Pull) ProtoMessage() {}

func (x *Pull) ProtoReflect() protoreflect.Message {
	mi := &file_ui_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Pull.ProtoReflect.Descriptor instead.
func (*Pull) Descriptor() ([]byte, []int) {
	return file_ui_proto_rawDescGZIP(), []int{2}
}

func (x *Pull) GetActives() []uint32 {
	if x != nil {
		return x.Actives
	}
	return nil
}

// Panel registers a panel in the puller webworker.
type Panel struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Id            uint32                    `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	Paths         []*tree_go_proto.NodePath `protobuf:"bytes,2,rep,name=paths,proto3" json:"paths,omitempty"`
	Transferables []uint32                  `protobuf:"varint,3,rep,packed,name=transferables,proto3" json:"transferables,omitempty"`
	Renderer      string                    `protobuf:"bytes,4,opt,name=renderer,proto3" json:"renderer,omitempty"`
}

func (x *Panel) Reset() {
	*x = Panel{}
	if protoimpl.UnsafeEnabled {
		mi := &file_ui_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Panel) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Panel) ProtoMessage() {}

func (x *Panel) ProtoReflect() protoreflect.Message {
	mi := &file_ui_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Panel.ProtoReflect.Descriptor instead.
func (*Panel) Descriptor() ([]byte, []int) {
	return file_ui_proto_rawDescGZIP(), []int{3}
}

func (x *Panel) GetId() uint32 {
	if x != nil {
		return x.Id
	}
	return 0
}

func (x *Panel) GetPaths() []*tree_go_proto.NodePath {
	if x != nil {
		return x.Paths
	}
	return nil
}

func (x *Panel) GetTransferables() []uint32 {
	if x != nil {
		return x.Transferables
	}
	return nil
}

func (x *Panel) GetRenderer() string {
	if x != nil {
		return x.Renderer
	}
	return ""
}

// StyleChange is sent to the workers to change the theme.
type StyleChange struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Theme      string  `protobuf:"bytes,1,opt,name=theme,proto3" json:"theme,omitempty"`
	FontFamily string  `protobuf:"bytes,2,opt,name=fontFamily,proto3" json:"fontFamily,omitempty"`
	FontSize   float64 `protobuf:"fixed64,3,opt,name=fontSize,proto3" json:"fontSize,omitempty"`
}

func (x *StyleChange) Reset() {
	*x = StyleChange{}
	if protoimpl.UnsafeEnabled {
		mi := &file_ui_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *StyleChange) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*StyleChange) ProtoMessage() {}

func (x *StyleChange) ProtoReflect() protoreflect.Message {
	mi := &file_ui_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use StyleChange.ProtoReflect.Descriptor instead.
func (*StyleChange) Descriptor() ([]byte, []int) {
	return file_ui_proto_rawDescGZIP(), []int{4}
}

func (x *StyleChange) GetTheme() string {
	if x != nil {
		return x.Theme
	}
	return ""
}

func (x *StyleChange) GetFontFamily() string {
	if x != nil {
		return x.FontFamily
	}
	return ""
}

func (x *StyleChange) GetFontSize() float64 {
	if x != nil {
		return x.FontSize
	}
	return 0
}

// ParentResize is sent when the parent of a renderer is changing its size.
type ParentResize struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	PanelID     uint32 `protobuf:"varint,1,opt,name=panelID,proto3" json:"panelID,omitempty"`
	ChildWidth  int32  `protobuf:"varint,10,opt,name=childWidth,proto3" json:"childWidth,omitempty"`
	ChildHeight int32  `protobuf:"varint,11,opt,name=childHeight,proto3" json:"childHeight,omitempty"`
}

func (x *ParentResize) Reset() {
	*x = ParentResize{}
	if protoimpl.UnsafeEnabled {
		mi := &file_ui_proto_msgTypes[5]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ParentResize) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ParentResize) ProtoMessage() {}

func (x *ParentResize) ProtoReflect() protoreflect.Message {
	mi := &file_ui_proto_msgTypes[5]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ParentResize.ProtoReflect.Descriptor instead.
func (*ParentResize) Descriptor() ([]byte, []int) {
	return file_ui_proto_rawDescGZIP(), []int{5}
}

func (x *ParentResize) GetPanelID() uint32 {
	if x != nil {
		return x.PanelID
	}
	return 0
}

func (x *ParentResize) GetChildWidth() int32 {
	if x != nil {
		return x.ChildWidth
	}
	return 0
}

func (x *ParentResize) GetChildHeight() int32 {
	if x != nil {
		return x.ChildHeight
	}
	return 0
}

// Event dispatched from the main thread to the renderers.
type UIEvent struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Types that are assignable to Event:
	//	*UIEvent_Style
	//	*UIEvent_Resize
	Event isUIEvent_Event `protobuf_oneof:"event"`
}

func (x *UIEvent) Reset() {
	*x = UIEvent{}
	if protoimpl.UnsafeEnabled {
		mi := &file_ui_proto_msgTypes[6]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *UIEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*UIEvent) ProtoMessage() {}

func (x *UIEvent) ProtoReflect() protoreflect.Message {
	mi := &file_ui_proto_msgTypes[6]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use UIEvent.ProtoReflect.Descriptor instead.
func (*UIEvent) Descriptor() ([]byte, []int) {
	return file_ui_proto_rawDescGZIP(), []int{6}
}

func (m *UIEvent) GetEvent() isUIEvent_Event {
	if m != nil {
		return m.Event
	}
	return nil
}

func (x *UIEvent) GetStyle() *StyleChange {
	if x, ok := x.GetEvent().(*UIEvent_Style); ok {
		return x.Style
	}
	return nil
}

func (x *UIEvent) GetResize() *ParentResize {
	if x, ok := x.GetEvent().(*UIEvent_Resize); ok {
		return x.Resize
	}
	return nil
}

type isUIEvent_Event interface {
	isUIEvent_Event()
}

type UIEvent_Style struct {
	Style *StyleChange `protobuf:"bytes,10,opt,name=style,proto3,oneof"`
}

type UIEvent_Resize struct {
	Resize *ParentResize `protobuf:"bytes,11,opt,name=resize,proto3,oneof"`
}

func (*UIEvent_Style) isUIEvent_Event() {}

func (*UIEvent_Resize) isUIEvent_Event() {}

// ToPuller is the message sent to the puller webworker.
type ToPuller struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Types that are assignable to Query:
	//	*ToPuller_Pull
	//	*ToPuller_RegisterPanel
	//	*ToPuller_UnregisterPanel
	//	*ToPuller_Event
	Query isToPuller_Query `protobuf_oneof:"query"`
}

func (x *ToPuller) Reset() {
	*x = ToPuller{}
	if protoimpl.UnsafeEnabled {
		mi := &file_ui_proto_msgTypes[7]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ToPuller) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ToPuller) ProtoMessage() {}

func (x *ToPuller) ProtoReflect() protoreflect.Message {
	mi := &file_ui_proto_msgTypes[7]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ToPuller.ProtoReflect.Descriptor instead.
func (*ToPuller) Descriptor() ([]byte, []int) {
	return file_ui_proto_rawDescGZIP(), []int{7}
}

func (m *ToPuller) GetQuery() isToPuller_Query {
	if m != nil {
		return m.Query
	}
	return nil
}

func (x *ToPuller) GetPull() *Pull {
	if x, ok := x.GetQuery().(*ToPuller_Pull); ok {
		return x.Pull
	}
	return nil
}

func (x *ToPuller) GetRegisterPanel() *Panel {
	if x, ok := x.GetQuery().(*ToPuller_RegisterPanel); ok {
		return x.RegisterPanel
	}
	return nil
}

func (x *ToPuller) GetUnregisterPanel() *Panel {
	if x, ok := x.GetQuery().(*ToPuller_UnregisterPanel); ok {
		return x.UnregisterPanel
	}
	return nil
}

func (x *ToPuller) GetEvent() *UIEvent {
	if x, ok := x.GetQuery().(*ToPuller_Event); ok {
		return x.Event
	}
	return nil
}

type isToPuller_Query interface {
	isToPuller_Query()
}

type ToPuller_Pull struct {
	Pull *Pull `protobuf:"bytes,1,opt,name=pull,proto3,oneof"`
}

type ToPuller_RegisterPanel struct {
	RegisterPanel *Panel `protobuf:"bytes,2,opt,name=registerPanel,proto3,oneof"`
}

type ToPuller_UnregisterPanel struct {
	UnregisterPanel *Panel `protobuf:"bytes,3,opt,name=unregisterPanel,proto3,oneof"`
}

type ToPuller_Event struct {
	Event *UIEvent `protobuf:"bytes,4,opt,name=event,proto3,oneof"`
}

func (*ToPuller_Pull) isToPuller_Query() {}

func (*ToPuller_RegisterPanel) isToPuller_Query() {}

func (*ToPuller_UnregisterPanel) isToPuller_Query() {}

func (*ToPuller_Event) isToPuller_Query() {}

// Data for a panel.
type PanelData struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Nodes []*tree_go_proto.NodeData `protobuf:"bytes,1,rep,name=nodes,proto3" json:"nodes,omitempty"`
}

func (x *PanelData) Reset() {
	*x = PanelData{}
	if protoimpl.UnsafeEnabled {
		mi := &file_ui_proto_msgTypes[8]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *PanelData) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PanelData) ProtoMessage() {}

func (x *PanelData) ProtoReflect() protoreflect.Message {
	mi := &file_ui_proto_msgTypes[8]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PanelData.ProtoReflect.Descriptor instead.
func (*PanelData) Descriptor() ([]byte, []int) {
	return file_ui_proto_rawDescGZIP(), []int{8}
}

func (x *PanelData) GetNodes() []*tree_go_proto.NodeData {
	if x != nil {
		return x.Nodes
	}
	return nil
}

// Message storing the data to be displayed by the main thread.
type DisplayData struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Err  string                `protobuf:"bytes,1,opt,name=err,proto3" json:"err,omitempty"`
	Data map[uint32]*PanelData `protobuf:"bytes,2,rep,name=data,proto3" json:"data,omitempty" protobuf_key:"varint,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
}

func (x *DisplayData) Reset() {
	*x = DisplayData{}
	if protoimpl.UnsafeEnabled {
		mi := &file_ui_proto_msgTypes[9]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DisplayData) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DisplayData) ProtoMessage() {}

func (x *DisplayData) ProtoReflect() protoreflect.Message {
	mi := &file_ui_proto_msgTypes[9]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DisplayData.ProtoReflect.Descriptor instead.
func (*DisplayData) Descriptor() ([]byte, []int) {
	return file_ui_proto_rawDescGZIP(), []int{9}
}

func (x *DisplayData) GetErr() string {
	if x != nil {
		return x.Err
	}
	return ""
}

func (x *DisplayData) GetData() map[uint32]*PanelData {
	if x != nil {
		return x.Data
	}
	return nil
}

var File_ui_proto protoreflect.FileDescriptor

var file_ui_proto_rawDesc = []byte{
	0x0a, 0x08, 0x75, 0x69, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0a, 0x6d, 0x75, 0x6c, 0x74,
	0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x1a, 0x0a, 0x74, 0x72, 0x65, 0x65, 0x2e, 0x70, 0x72, 0x6f,
	0x74, 0x6f, 0x22, 0x0b, 0x0a, 0x09, 0x57, 0x6f, 0x72, 0x6b, 0x65, 0x72, 0x41, 0x63, 0x6b, 0x22,
	0x35, 0x0a, 0x07, 0x43, 0x6f, 0x6e, 0x6e, 0x65, 0x63, 0x74, 0x12, 0x16, 0x0a, 0x06, 0x73, 0x63,
	0x68, 0x65, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x73, 0x63, 0x68, 0x65,
	0x6d, 0x65, 0x12, 0x12, 0x0a, 0x04, 0x68, 0x6f, 0x73, 0x74, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x04, 0x68, 0x6f, 0x73, 0x74, 0x22, 0x20, 0x0a, 0x04, 0x50, 0x75, 0x6c, 0x6c, 0x12, 0x18,
	0x0a, 0x07, 0x61, 0x63, 0x74, 0x69, 0x76, 0x65, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0d, 0x52,
	0x07, 0x61, 0x63, 0x74, 0x69, 0x76, 0x65, 0x73, 0x22, 0x85, 0x01, 0x0a, 0x05, 0x50, 0x61, 0x6e,
	0x65, 0x6c, 0x12, 0x0e, 0x0a, 0x02, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x02,
	0x69, 0x64, 0x12, 0x2a, 0x0a, 0x05, 0x70, 0x61, 0x74, 0x68, 0x73, 0x18, 0x02, 0x20, 0x03, 0x28,
	0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x4e,
	0x6f, 0x64, 0x65, 0x50, 0x61, 0x74, 0x68, 0x52, 0x05, 0x70, 0x61, 0x74, 0x68, 0x73, 0x12, 0x24,
	0x0a, 0x0d, 0x74, 0x72, 0x61, 0x6e, 0x73, 0x66, 0x65, 0x72, 0x61, 0x62, 0x6c, 0x65, 0x73, 0x18,
	0x03, 0x20, 0x03, 0x28, 0x0d, 0x52, 0x0d, 0x74, 0x72, 0x61, 0x6e, 0x73, 0x66, 0x65, 0x72, 0x61,
	0x62, 0x6c, 0x65, 0x73, 0x12, 0x1a, 0x0a, 0x08, 0x72, 0x65, 0x6e, 0x64, 0x65, 0x72, 0x65, 0x72,
	0x18, 0x04, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x72, 0x65, 0x6e, 0x64, 0x65, 0x72, 0x65, 0x72,
	0x22, 0x5f, 0x0a, 0x0b, 0x53, 0x74, 0x79, 0x6c, 0x65, 0x43, 0x68, 0x61, 0x6e, 0x67, 0x65, 0x12,
	0x14, 0x0a, 0x05, 0x74, 0x68, 0x65, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05,
	0x74, 0x68, 0x65, 0x6d, 0x65, 0x12, 0x1e, 0x0a, 0x0a, 0x66, 0x6f, 0x6e, 0x74, 0x46, 0x61, 0x6d,
	0x69, 0x6c, 0x79, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0a, 0x66, 0x6f, 0x6e, 0x74, 0x46,
	0x61, 0x6d, 0x69, 0x6c, 0x79, 0x12, 0x1a, 0x0a, 0x08, 0x66, 0x6f, 0x6e, 0x74, 0x53, 0x69, 0x7a,
	0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x01, 0x52, 0x08, 0x66, 0x6f, 0x6e, 0x74, 0x53, 0x69, 0x7a,
	0x65, 0x22, 0x6a, 0x0a, 0x0c, 0x50, 0x61, 0x72, 0x65, 0x6e, 0x74, 0x52, 0x65, 0x73, 0x69, 0x7a,
	0x65, 0x12, 0x18, 0x0a, 0x07, 0x70, 0x61, 0x6e, 0x65, 0x6c, 0x49, 0x44, 0x18, 0x01, 0x20, 0x01,
	0x28, 0x0d, 0x52, 0x07, 0x70, 0x61, 0x6e, 0x65, 0x6c, 0x49, 0x44, 0x12, 0x1e, 0x0a, 0x0a, 0x63,
	0x68, 0x69, 0x6c, 0x64, 0x57, 0x69, 0x64, 0x74, 0x68, 0x18, 0x0a, 0x20, 0x01, 0x28, 0x05, 0x52,
	0x0a, 0x63, 0x68, 0x69, 0x6c, 0x64, 0x57, 0x69, 0x64, 0x74, 0x68, 0x12, 0x20, 0x0a, 0x0b, 0x63,
	0x68, 0x69, 0x6c, 0x64, 0x48, 0x65, 0x69, 0x67, 0x68, 0x74, 0x18, 0x0b, 0x20, 0x01, 0x28, 0x05,
	0x52, 0x0b, 0x63, 0x68, 0x69, 0x6c, 0x64, 0x48, 0x65, 0x69, 0x67, 0x68, 0x74, 0x22, 0x77, 0x0a,
	0x07, 0x55, 0x49, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x12, 0x2f, 0x0a, 0x05, 0x73, 0x74, 0x79, 0x6c,
	0x65, 0x18, 0x0a, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73,
	0x63, 0x6f, 0x70, 0x65, 0x2e, 0x53, 0x74, 0x79, 0x6c, 0x65, 0x43, 0x68, 0x61, 0x6e, 0x67, 0x65,
	0x48, 0x00, 0x52, 0x05, 0x73, 0x74, 0x79, 0x6c, 0x65, 0x12, 0x32, 0x0a, 0x06, 0x72, 0x65, 0x73,
	0x69, 0x7a, 0x65, 0x18, 0x0b, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x18, 0x2e, 0x6d, 0x75, 0x6c, 0x74,
	0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x50, 0x61, 0x72, 0x65, 0x6e, 0x74, 0x52, 0x65, 0x73,
	0x69, 0x7a, 0x65, 0x48, 0x00, 0x52, 0x06, 0x72, 0x65, 0x73, 0x69, 0x7a, 0x65, 0x42, 0x07, 0x0a,
	0x05, 0x65, 0x76, 0x65, 0x6e, 0x74, 0x22, 0xe2, 0x01, 0x0a, 0x08, 0x54, 0x6f, 0x50, 0x75, 0x6c,
	0x6c, 0x65, 0x72, 0x12, 0x26, 0x0a, 0x04, 0x70, 0x75, 0x6c, 0x6c, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x0b, 0x32, 0x10, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x50,
	0x75, 0x6c, 0x6c, 0x48, 0x00, 0x52, 0x04, 0x70, 0x75, 0x6c, 0x6c, 0x12, 0x39, 0x0a, 0x0d, 0x72,
	0x65, 0x67, 0x69, 0x73, 0x74, 0x65, 0x72, 0x50, 0x61, 0x6e, 0x65, 0x6c, 0x18, 0x02, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x11, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e,
	0x50, 0x61, 0x6e, 0x65, 0x6c, 0x48, 0x00, 0x52, 0x0d, 0x72, 0x65, 0x67, 0x69, 0x73, 0x74, 0x65,
	0x72, 0x50, 0x61, 0x6e, 0x65, 0x6c, 0x12, 0x3d, 0x0a, 0x0f, 0x75, 0x6e, 0x72, 0x65, 0x67, 0x69,
	0x73, 0x74, 0x65, 0x72, 0x50, 0x61, 0x6e, 0x65, 0x6c, 0x18, 0x03, 0x20, 0x01, 0x28, 0x0b, 0x32,
	0x11, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x50, 0x61, 0x6e,
	0x65, 0x6c, 0x48, 0x00, 0x52, 0x0f, 0x75, 0x6e, 0x72, 0x65, 0x67, 0x69, 0x73, 0x74, 0x65, 0x72,
	0x50, 0x61, 0x6e, 0x65, 0x6c, 0x12, 0x2b, 0x0a, 0x05, 0x65, 0x76, 0x65, 0x6e, 0x74, 0x18, 0x04,
	0x20, 0x01, 0x28, 0x0b, 0x32, 0x13, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70,
	0x65, 0x2e, 0x55, 0x49, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x48, 0x00, 0x52, 0x05, 0x65, 0x76, 0x65,
	0x6e, 0x74, 0x42, 0x07, 0x0a, 0x05, 0x71, 0x75, 0x65, 0x72, 0x79, 0x22, 0x37, 0x0a, 0x09, 0x50,
	0x61, 0x6e, 0x65, 0x6c, 0x44, 0x61, 0x74, 0x61, 0x12, 0x2a, 0x0a, 0x05, 0x6e, 0x6f, 0x64, 0x65,
	0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73,
	0x63, 0x6f, 0x70, 0x65, 0x2e, 0x4e, 0x6f, 0x64, 0x65, 0x44, 0x61, 0x74, 0x61, 0x52, 0x05, 0x6e,
	0x6f, 0x64, 0x65, 0x73, 0x22, 0xa6, 0x01, 0x0a, 0x0b, 0x44, 0x69, 0x73, 0x70, 0x6c, 0x61, 0x79,
	0x44, 0x61, 0x74, 0x61, 0x12, 0x10, 0x0a, 0x03, 0x65, 0x72, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x03, 0x65, 0x72, 0x72, 0x12, 0x35, 0x0a, 0x04, 0x64, 0x61, 0x74, 0x61, 0x18, 0x02,
	0x20, 0x03, 0x28, 0x0b, 0x32, 0x21, 0x2e, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70,
	0x65, 0x2e, 0x44, 0x69, 0x73, 0x70, 0x6c, 0x61, 0x79, 0x44, 0x61, 0x74, 0x61, 0x2e, 0x44, 0x61,
	0x74, 0x61, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x52, 0x04, 0x64, 0x61, 0x74, 0x61, 0x1a, 0x4e, 0x0a,
	0x09, 0x44, 0x61, 0x74, 0x61, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x12, 0x10, 0x0a, 0x03, 0x6b, 0x65,
	0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x03, 0x6b, 0x65, 0x79, 0x12, 0x2b, 0x0a, 0x05,
	0x76, 0x61, 0x6c, 0x75, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x15, 0x2e, 0x6d, 0x75,
	0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2e, 0x50, 0x61, 0x6e, 0x65, 0x6c, 0x44, 0x61,
	0x74, 0x61, 0x52, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x3a, 0x02, 0x38, 0x01, 0x42, 0x1f, 0x5a,
	0x1d, 0x6d, 0x75, 0x6c, 0x74, 0x69, 0x73, 0x63, 0x6f, 0x70, 0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x73, 0x2f, 0x75, 0x69, 0x5f, 0x67, 0x6f, 0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x06,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_ui_proto_rawDescOnce sync.Once
	file_ui_proto_rawDescData = file_ui_proto_rawDesc
)

func file_ui_proto_rawDescGZIP() []byte {
	file_ui_proto_rawDescOnce.Do(func() {
		file_ui_proto_rawDescData = protoimpl.X.CompressGZIP(file_ui_proto_rawDescData)
	})
	return file_ui_proto_rawDescData
}

var file_ui_proto_msgTypes = make([]protoimpl.MessageInfo, 11)
var file_ui_proto_goTypes = []interface{}{
	(*WorkerAck)(nil),              // 0: multiscope.WorkerAck
	(*Connect)(nil),                // 1: multiscope.Connect
	(*Pull)(nil),                   // 2: multiscope.Pull
	(*Panel)(nil),                  // 3: multiscope.Panel
	(*StyleChange)(nil),            // 4: multiscope.StyleChange
	(*ParentResize)(nil),           // 5: multiscope.ParentResize
	(*UIEvent)(nil),                // 6: multiscope.UIEvent
	(*ToPuller)(nil),               // 7: multiscope.ToPuller
	(*PanelData)(nil),              // 8: multiscope.PanelData
	(*DisplayData)(nil),            // 9: multiscope.DisplayData
	nil,                            // 10: multiscope.DisplayData.DataEntry
	(*tree_go_proto.NodePath)(nil), // 11: multiscope.NodePath
	(*tree_go_proto.NodeData)(nil), // 12: multiscope.NodeData
}
var file_ui_proto_depIdxs = []int32{
	11, // 0: multiscope.Panel.paths:type_name -> multiscope.NodePath
	4,  // 1: multiscope.UIEvent.style:type_name -> multiscope.StyleChange
	5,  // 2: multiscope.UIEvent.resize:type_name -> multiscope.ParentResize
	2,  // 3: multiscope.ToPuller.pull:type_name -> multiscope.Pull
	3,  // 4: multiscope.ToPuller.registerPanel:type_name -> multiscope.Panel
	3,  // 5: multiscope.ToPuller.unregisterPanel:type_name -> multiscope.Panel
	6,  // 6: multiscope.ToPuller.event:type_name -> multiscope.UIEvent
	12, // 7: multiscope.PanelData.nodes:type_name -> multiscope.NodeData
	10, // 8: multiscope.DisplayData.data:type_name -> multiscope.DisplayData.DataEntry
	8,  // 9: multiscope.DisplayData.DataEntry.value:type_name -> multiscope.PanelData
	10, // [10:10] is the sub-list for method output_type
	10, // [10:10] is the sub-list for method input_type
	10, // [10:10] is the sub-list for extension type_name
	10, // [10:10] is the sub-list for extension extendee
	0,  // [0:10] is the sub-list for field type_name
}

func init() { file_ui_proto_init() }
func file_ui_proto_init() {
	if File_ui_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_ui_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*WorkerAck); i {
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
		file_ui_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Connect); i {
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
		file_ui_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Pull); i {
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
		file_ui_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Panel); i {
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
		file_ui_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*StyleChange); i {
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
		file_ui_proto_msgTypes[5].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ParentResize); i {
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
		file_ui_proto_msgTypes[6].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*UIEvent); i {
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
		file_ui_proto_msgTypes[7].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ToPuller); i {
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
		file_ui_proto_msgTypes[8].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*PanelData); i {
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
		file_ui_proto_msgTypes[9].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*DisplayData); i {
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
	file_ui_proto_msgTypes[6].OneofWrappers = []interface{}{
		(*UIEvent_Style)(nil),
		(*UIEvent_Resize)(nil),
	}
	file_ui_proto_msgTypes[7].OneofWrappers = []interface{}{
		(*ToPuller_Pull)(nil),
		(*ToPuller_RegisterPanel)(nil),
		(*ToPuller_UnregisterPanel)(nil),
		(*ToPuller_Event)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_ui_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   11,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_ui_proto_goTypes,
		DependencyIndexes: file_ui_proto_depIdxs,
		MessageInfos:      file_ui_proto_msgTypes,
	}.Build()
	File_ui_proto = out.File
	file_ui_proto_rawDesc = nil
	file_ui_proto_goTypes = nil
	file_ui_proto_depIdxs = nil
}
