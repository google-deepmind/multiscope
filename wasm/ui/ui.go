// Package ui provides core UI abstractions.
package ui

import (
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/renderers"
	"multiscope/wasm/settings"
	"syscall/js"

	"google.golang.org/protobuf/proto"
	"honnef.co/go/js/dom/v2"
)

type (
	// PanelID of a panel for worker communication.
	PanelID uint64

	// UI is top structures owning all the UI elements.
	UI interface {
		// Owner of the UI.
		Owner() *Owner

		// Settings returns the global settings.
		Settings() *settings.Settings

		// TreeClient returns the connection to the server.
		TreeClient() treepb.TreeClient

		// Dashboard returns the main dashboard displaying the panels.
		Dashboard() Dashboard

		// DisplayErr displays an error for the user to see.
		DisplayErr(err error)

		// Run a function in the background.
		// Use this function to process an event and to avoid a deadlock
		// when the call includes gRPC calls.
		Run(func() error)
	}

	// Dashboard displaying all the panels.
	Dashboard interface {
		UI() UI

		NewDescriptor(node *treepb.Node, renderer renderers.Newer, paths ...*treepb.NodePath) Descriptor

		OpenPanel(node *treepb.Node) error

		ClosePanel(pnl Panel) error
	}

	// Descriptor enables the communication between a panel and the web worker to get the data.
	Descriptor interface {
		// Path returns the path of the node that the descriptor stores data about.
		// Returns nil if the descriptor does correspond to a path in the tree.
		Path() *treepb.NodePath

		// Dashboard returns the owner of the descriptor.
		Dashboard() Dashboard

		// AddTransferable adds a Javascript to transfer to the renderer of the web-worker.
		AddTransferable(name string, v js.Value)
	}

	// Panel is a display within the dashboard.
	Panel interface {
		// Root returns the root node of a panel.
		// This node is added to the dashboard node when a panel is registered.
		Root() *dom.HTMLDivElement
		// Desc returns the panel descriptor.
		Desc() Descriptor
		// Display the latest data.
		Display(node *treepb.NodeData)
	}

	// PanelBuilder builds a display given a node in the tree.
	PanelBuilder func(dbd Dashboard, node *treepb.Node) (Panel, error)
)

var mimeToDisplay = make(map[string]PanelBuilder)

// RegisterBuilder registers a builder given a mime type.
func RegisterBuilder(mime string, f PanelBuilder) {
	mimeToDisplay[mime] = f
}

// RegisterBuilderPB registers a builder given a protocol buffer type.
func RegisterBuilderPB(msg proto.Message, f PanelBuilder) {
	RegisterBuilder("application/x-protobuf;proto="+string(proto.MessageName(msg)), f)
}

// Builder returns the registered builder for a given MIME type
// (or nil if no builder has been registered).
func Builder(mime string) PanelBuilder {
	return mimeToDisplay[mime]
}
