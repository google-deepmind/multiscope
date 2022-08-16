package ui

import (
	"multiscope/internal/mime"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/renderers"
	"syscall/js"

	"github.com/pkg/errors"
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

	// Descriptor stores how to get and process data for a panel.
	Descriptor struct {
		dbd           *Dashboard
		pb            uipb.Panel
		transferables map[string]any
	}

	// Panel is a display within the dashboard.
	Panel interface {
		// Root returns the root node of a panel.
		// This node is added to the dashboard node when a panel is registered.
		Root() dom.Node
		// Desc returns the panel descriptor.
		Desc() *Descriptor
		// Display the latest data.
		Display(node *treepb.NodeData)
	}

	// DisplayerBuilder builds a display given a node in the tree.
	DisplayerBuilder func(dbd *Dashboard, node *treepb.Node) (Panel, error)
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
func (dbd *Dashboard) NewDescriptor(renderer renderers.Newer, paths ...*treepb.NodePath) *Descriptor {
	id := nextID
	nextID++
	return &Descriptor{
		dbd: dbd,
		pb: uipb.Panel{
			Id:       id,
			Paths:    paths,
			Renderer: renderers.Name(renderer),
		}}
}

// AddTransferable adds a key,value pair to transfer to the renderer worker.
func (dsc *Descriptor) AddTransferable(name string, v js.Value) {
	if dsc.transferables == nil {
		dsc.transferables = make(map[string]any)
	}
	dsc.transferables[name] = v
}

// ID returns the ID of the panel.
func (dsc *Descriptor) ID() PanelID {
	return PanelID(dsc.pb.Id)
}

// PanelPB returns the list of path necessary for the panel.
func (dsc *Descriptor) PanelPB() (*uipb.Panel, map[string]any) {
	return &dsc.pb, dsc.transferables
}

// Dashboard returns the owner of the descriptor.
func (dsc *Descriptor) Dashboard() *Dashboard {
	return dsc.dbd
}

func (dbd *Dashboard) buildPanel(node *treepb.Node) (Panel, error) {
	builder := mimeToDisplay[node.Mime]
	if builder == nil {
		builder = mimeToDisplay[mime.Unsupported]
	}
	if builder == nil {
		return nil, errors.Errorf("MIME type %q not supported", mime.Unsupported)
	}
	return builder(dbd, node)
}
