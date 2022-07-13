package ui

import (
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/renderers"
	"syscall/js"

	"google.golang.org/protobuf/proto"
	"honnef.co/go/js/dom/v2"
)

type (
	// PanelID of a panel for worker communication.
	PanelID uint64

	// Parent in the dom tree to which a panel can be added.
	Parent interface {
		AppendChild(dom.Node)
	}

	// Displayer on the display.
	Displayer interface {
		Display(*treepb.NodeData) error
	}

	// DisplayerBuilder builds a display given a node in the tree.
	DisplayerBuilder func(dbd *Dashboard, node *treepb.Node) (*Panel, error)

	// Descriptor stores how to get and process data for a panel.
	Descriptor struct {
		pb            uipb.Panel
		transferables map[string]any
		disp          Displayer
		root          dom.HTMLElement
	}
)

var (
	nextID        = uint32(1)
	mimeToDisplay = make(map[string]DisplayerBuilder)
)

// RegisterDisplay registers a display given a mime type.
func RegisterDisplay(mime string, f DisplayerBuilder) {
	mimeToDisplay[mime] = f
}

// RegisterDisplayPB registers a display given a protocol buffer type.
func RegisterDisplayPB(msg proto.Message, f DisplayerBuilder) {
	RegisterDisplay("application/x-protobuf;proto="+string(proto.MessageName(msg)), f)
}

func rootDescriptor() *Descriptor {
	return &Descriptor{
		pb: uipb.Panel{
			Id:    0,
			Paths: []*treepb.NodePath{&treepb.NodePath{}},
		}}
}

// NewDescriptor returns a new info descriptor to assign to a panel.
func NewDescriptor(disp Displayer, root dom.HTMLElement, renderer renderers.Newer, paths ...*treepb.NodePath) *Descriptor {
	id := nextID
	nextID++
	return &Descriptor{
		disp: disp,
		root: root,
		pb: uipb.Panel{
			Id:       id,
			Paths:    paths,
			Renderer: renderers.Name(renderer),
		}}
}

// AddTransferable adds a key,value pair to transfer to the renderer worker.
func (inf *Descriptor) AddTransferable(name string, v js.Value) {
	if inf.transferables == nil {
		inf.transferables = make(map[string]any)
	}
	inf.transferables[name] = v
}

// ID returns the ID of the panel.
func (inf *Descriptor) ID() PanelID {
	return PanelID(inf.pb.Id)
}

// PanelPB returns the list of path necessary for the panel.
func (inf *Descriptor) PanelPB() (*uipb.Panel, map[string]any) {
	return &inf.pb, inf.transferables
}

func (dbd *Dashboard) buildPanel(node *treepb.Node) (*Panel, error) {
	builder := mimeToDisplay[node.Mime]
	if builder == nil {
		return newUnsupported(dbd, node)
	}
	return builder(dbd, node)
}
